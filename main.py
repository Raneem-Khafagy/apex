"""
APEX Runtime Daemon — async entry point.

Wires the full pipeline and runs it in a single asyncio event loop:

    ProductivityAdapter
         │
    SignalMonitor ──────────────────► PipelineCoordinator ──► ContextBuffer
         │                                                          │
         │  (heartbeat + watchfiles events)                         │
         │                                                          ▼
    TauCalibrator ◄── DuckDB ◄── log_prefetch()            FastAPI MCP Server
                                                           (uvicorn on :8765)

Usage
-----
    # Prerequisites: Ollama running + index built
    ollama serve &                   # separate terminal (just llm)
    just ingest                      # first time, or after vault changes
    uv run python main.py            # start the daemon (just dev)

Environment variables
---------------------
    APEX_VAULT_PATH     Path to the knowledge base directory.
                        Default: ~/Documents/ApexVault
    APEX_INDEX_PATH     Base path for saved HNSW + metadata index.
                        Default: ./apex_vault
    APEX_DB_PATH        DuckDB evaluation store file.
                        Default: ./apex_eval.db
    APEX_PORT           FastAPI port.
                        Default: 8765
    APEX_LOG_LEVEL      Loguru level (DEBUG, INFO, WARNING).
                        Default: INFO
    APEX_TAU_FIXED      Disable τ calibration (1 = fixed at 0.65, 0 = adaptive).
                        Default: 0 (adaptive calibration enabled)

Privacy: no document content is logged or transmitted.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

import uvicorn
from loguru import logger

from apex.adapter.llm_adapter import LLMAdapter
from apex.adapters.factory import FactoryAdapter
from apex.adapters.productivity import ProductivityAdapter
from apex.analytics.store import AnalyticsStore
from apex.buffer.context_buffer import ContextBuffer
from apex.inference.intent_engine import IntentEngine
from apex.inference.tau_calibrator import TauCalibrator
from apex.ingest.ingestor import Ingestor
from apex.monitor.live import LiveDisplay
from apex.monitor.signal_monitor import SignalMonitor
from apex.pipeline.coordinator import PipelineCoordinator
from apex.retrieval.rrf import RetrievalEngine
from apex.scheduler.speculative import SpeculativeScheduler
import apex.server as _server

# IEEE IoT τ persistence (cross-session τ convergence measurement)
_IEEE_IOT_TAU = None
try:
    import importlib, pathlib
    _tau_mod_path = pathlib.Path(__file__).parent.parent / "ieee_iot" / "tau" / "tau_persistence.py"
    if _tau_mod_path.exists():
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("tau_persistence", _tau_mod_path)
        _IEEE_IOT_TAU = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_IEEE_IOT_TAU)
except Exception:
    pass  # optional component — safe to skip if ieee_iot is not on path


# ── Config from environment ────────────────────────────────────────────────────

def _config() -> dict:
    return {
        "vault_path":  os.path.expanduser(
            os.environ.get("APEX_VAULT_PATH", "~/Documents/ApexVault")
        ),
        "index_path":  os.environ.get("APEX_INDEX_PATH", "apex_vault"),
        "db_path":     os.environ.get("APEX_DB_PATH", "apex_eval.db"),
        "port":        int(os.environ.get("APEX_PORT", "8765")),
        "log_level":   os.environ.get("APEX_LOG_LEVEL", "INFO"),
        # Model selection — override to swap models without touching pipeline code.
        # Example: APEX_LLM_MODEL=llama3.2:1b just dev   (smaller, faster)
        "llm_model":   os.environ.get("APEX_LLM_MODEL",   "phi3.5"),
        "embed_model": os.environ.get("APEX_EMBED_MODEL",  "all-minilm"),
        # APEX_ADAPTER: "productivity" (default) | "factory"
        # APEX_SENSOR_STATE_PATH: path to factory JSON state file (factory adapter only)
        "adapter_type":       os.environ.get("APEX_ADAPTER", "productivity"),
        "sensor_state_path":  os.environ.get("APEX_SENSOR_STATE_PATH",
                                             "/tmp/apex_factory_state.json"),
    }


def _configure_logging(level: str) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, colorize=True,
               format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")


# ── Index loading ──────────────────────────────────────────────────────────────

def _load_or_warn_index(engine: RetrievalEngine, index_path: str) -> None:
    """
    Load the pre-built HNSW index if it exists.

    If the index is missing, log a clear warning and continue with an
    empty engine — retrievals will return nothing until `just ingest` runs.
    """
    hnsw_file = index_path + ".hnsw"
    meta_file = index_path + ".meta.json"

    if os.path.exists(hnsw_file) and os.path.exists(meta_file):
        ingestor = Ingestor(engine)
        ingestor.load_index(index_path)
        logger.info("main: index loaded from '{}' ({} chunks)",
                    index_path, len(ingestor._metadata))
    else:
        logger.warning(
            "main: no index found at '{}'. "
            "Retrieval will return empty results until you run: just ingest",
            index_path,
        )


# ── Main coroutine ─────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="APEX Runtime Daemon — proactive context delivery system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--fixed-tau",
        action="store_true",
        help="Disable τ calibration (fixed at 0.65 for baseline comparison). "
             "Equivalent to APEX_TAU_FIXED=1"
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    cfg = _config()
    _configure_logging(cfg["log_level"])

    # Override environment variable if command line flag is provided
    if args.fixed_tau:
        os.environ["APEX_TAU_FIXED"] = "1"

    session_id = os.environ.get("APEX_SESSION_ID") or str(uuid.uuid4())
    logger.info("APEX daemon starting — vault='{}' port={} session={}",
                cfg["vault_path"], cfg["port"], session_id)

    # ── [1] Build shared pipeline components ─────────────────────────────────
    engine     = RetrievalEngine()
    buffer     = ContextBuffer()
    store      = AnalyticsStore(db_path=cfg["db_path"])
    iie        = IntentEngine(chat_model=cfg["llm_model"], embed_model=cfg["embed_model"])
    scheduler  = SpeculativeScheduler()
    adapter    = LLMAdapter(model=cfg["llm_model"])

    # ── [2] Load knowledge base index ────────────────────────────────────────
    _load_or_warn_index(engine, cfg["index_path"])

    # ── [3] Build terminal display (screen=False: logs still visible alongside) ─
    display = LiveDisplay(refresh_rate=4.0, screen=False)

    # ── [4] Build coordinator ─────────────────────────────────────────────────
    coordinator = PipelineCoordinator(
        retrieval_engine=engine,
        buffer=buffer,
        intent_engine=iie,
        scheduler=scheduler,
        store=store,
        session_id=session_id,
        display=display,
    )

    # ── [5] Inject into FastAPI server (also wires push_callback + sse_broadcast) ─
    _server.init_app(
        buffer=buffer,
        adapter=adapter,
        coordinator=coordinator,
        engine=engine,
        vault_path=cfg["vault_path"],
        index_path=cfg["index_path"],
    )

    # ── [6] Build Behavioral Signal Monitor ───────────────────────────────────
    if cfg["adapter_type"] == "factory":
        signal_adapter = FactoryAdapter(sensor_state_path=cfg["sensor_state_path"])
        # Watch the state file directly — avoids watchfiles scanning /tmp root
        # which contains unreadable systemd-private-* directories.
        watch_path = cfg["sensor_state_path"]
        logger.info("main: using FactoryAdapter — sensor_state='{}'",
                    cfg["sensor_state_path"])
    else:
        signal_adapter = ProductivityAdapter(watch_path=cfg["vault_path"])
        watch_path = cfg["vault_path"]
    monitor = SignalMonitor(
        adapter=signal_adapter,
        watch_path=watch_path,
    )
    monitor.register_callback(coordinator.process_signal_all)
    _server._monitor_ref = monitor

    # ── [7a] Load τ checkpoint (cross-session convergence) ───────────────────
    if _IEEE_IOT_TAU is not None:
        _tau_ckpt = _IEEE_IOT_TAU.DEFAULT_CHECKPOINT
        _IEEE_IOT_TAU.load_tau_checkpoint(scheduler, _tau_ckpt)

    # ── [7b] Build TauCalibrator ───────────────────────────────────────────────
    calibrator = TauCalibrator(
        store=store,
        scheduler=scheduler,
        session_id=session_id,
        interval=120.0,   # calibrate every 2 minutes during a session
        min_events=10,
    )

    # ── [8] Start uvicorn ─────────────────────────────────────────────────────
    # Bind to localhost only by default — never expose over the network.
    # Set APEX_HOST=0.0.0.0 explicitly only if you need LAN access.
    host = os.environ.get("APEX_HOST", "127.0.0.1")
    uv_config = uvicorn.Config(
        _server.app,
        host=host,
        port=cfg["port"],
        log_level="warning",   # loguru handles APEX logs; suppress uvicorn noise
    )
    uv_server = uvicorn.Server(uv_config)

    logger.info(
        "APEX: all components ready — starting event loop "
        "(dashboard: http://localhost:{}/ | models: llm={} embed={})",
        cfg["port"], cfg["llm_model"], cfg["embed_model"],
    )

    # ── [9] Run everything concurrently inside the Live terminal display ──────
    try:
        with display:
            await asyncio.gather(
                uv_server.serve(),
                monitor.run(),
                calibrator.run(),
            )
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("APEX: shutting down")
        monitor.stop()
        calibrator.stop()
        # Save τ checkpoint for cross-session convergence measurement
        if _IEEE_IOT_TAU is not None:
            try:
                _IEEE_IOT_TAU.save_tau_checkpoint(scheduler, session_id, _IEEE_IOT_TAU.DEFAULT_CHECKPOINT)
                _IEEE_IOT_TAU.log_tau_snapshot(
                    store, session_id,
                    float(scheduler._tau),
                    {k: float(v) for k, v in scheduler._domain_tau.items()},
                )
            except Exception as _e:
                logger.warning("τ checkpoint save failed: {}", _e)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

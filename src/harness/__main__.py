"""Console entry point: ``harness`` (or ``python -m harness``)."""

import argparse

import uvicorn

from .api import DEFAULT_PORT, create_app


def main() -> None:
    # Só o que precisa existir antes do app entra por flag: onde ouvir, e qual
    # host.json ler. Todo o resto vem do próprio host.json, editável no painel.
    parser = argparse.ArgumentParser(prog="harness", description="Sobe o servidor do Meu Harness.")
    parser.add_argument("--host", default="127.0.0.1", help="interface de bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="porta HTTP")
    parser.add_argument("--host-config", default=None, help="caminho alternativo do host.json")
    parser.add_argument(
        "--setup",
        action="store_true",
        help="reabre o setup neste boot, mesmo com o host já configurado",
    )
    args = parser.parse_args()
    app = create_app(
        host_config_path=args.host_config,
        reopen_setup=args.setup,
        port=args.port,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

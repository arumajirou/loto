import os

from config_utils import build_arg_parser, default_ini_path, from_ini_and_args
from runner import PipelineRunner

def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    ini_path = default_ini_path(args.config, here)
    cfg = from_ini_and_args(ini_path, args)

    runner = PipelineRunner(cfg)
    runner.run()

if __name__ == "__main__":
    main()

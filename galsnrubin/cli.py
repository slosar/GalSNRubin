"""CLI: simulation plus reproducible figures and report tables."""
import argparse
from dataclasses import replace
from .config import Config
from .simulation import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',required=True,help='Trusted TRILEGAL pandas pickle')
    parser.add_argument('--output',default='outputs/baseline_moon')
    parser.add_argument('--config',help='JSON settings; omitted fields use defaults')
    parser.add_argument('--n',type=int,help='Override candidates per progenitor')
    args = parser.parse_args()
    config = Config.load(args.config) if args.config else Config()
    if args.n is not None:
        config = replace(config,n_candidates=args.n)
    results = run(args.catalog,args.output,config)
    from .products import make_products
    make_products(results,args.output,config)


if __name__ == '__main__':
    main()

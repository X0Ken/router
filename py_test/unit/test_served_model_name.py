import argparse

import pytest
from vllm_router.router_args import RouterArgs


@pytest.mark.parametrize("use_prefix", [False, True])
def test_served_model_name_cli(use_prefix):
    parser = argparse.ArgumentParser()
    RouterArgs.add_cli_args(parser, use_router_prefix=use_prefix)
    flag = "--router-served-model-name" if use_prefix else "--served-model-name"
    parsed = parser.parse_args([flag, "dsv4"])
    args = RouterArgs.from_cli_args(parsed, use_router_prefix=use_prefix)
    assert args.served_model_name == "dsv4"


def test_served_model_name_defaults_to_proxy():
    assert RouterArgs().served_model_name is None


def test_served_model_name_reaches_native_validation():
    from vllm_router.router import Router

    args = RouterArgs(
        worker_urls=["http://127.0.0.1:8900"],
        served_model_name=" ",
    )
    router = Router.from_args(args)
    with pytest.raises(ValueError, match="served_model_name must be non-empty"):
        router.start()

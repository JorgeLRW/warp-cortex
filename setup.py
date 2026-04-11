from setuptools import setup, find_packages

setup(
    name="warp_cortex",
    version="0.3.0",
    packages=find_packages(include=[
        "cortex_core",
        "cortex_core.*",
        "cortex_scripts",
        "cortex_scripts.*",
        "cortex_benchmarks",
        "cortex_benchmarks.*",
        "cortex_resources",
        "cortex_resources.*",
    ]),
    py_modules=[
        "cortex_engine",
    ],
    install_requires=[
        "torch",
        "transformers",
        "PyYAML",
    ],
    extras_require={
        "api": ["openai"],
        "benchmarks": ["datasets"],
    },
    package_data={"cortex_resources": ["agent_skills/*.json"]},
    entry_points={
        "console_scripts": [
            "warp-cortex-live=cortex_scripts.council_live:main",
            "warp-cortex-gsm8k=cortex_benchmarks.benchmark_cortex_gsm8k:main",
        ]
    },
)

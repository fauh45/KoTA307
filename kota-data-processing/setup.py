from setuptools import find_packages, setup

setup(
    name="kota_data_processing",
    packages=find_packages(exclude=["kota_data_processing_tests"]),
    install_requires=["dagster", "ShopifyAPI", "boto3"],
    extras_require={
        "dev": ["dagit", "pytest", "boto3-stubs[essential]", "commitizen"]
    },
)

import os

from dagster import load_assets_from_package_module, repository, with_resources

from kota_data_processing import assets
from kota_data_processing.resources import shopify_api, s3


@repository
def kota_data_processing():
    return [
        *with_resources(
            load_assets_from_package_module(assets),
            {
                "shopify": shopify_api.shopify_resources.configured(
                    {
                        "shop_url": os.getenv("SHOP_URL"),
                        "api_version": os.getenv("API_VERSION"),
                        "access_token": os.getenv("ACCESS_TOKEN"),
                        "api_key": os.getenv("API_KEY"),
                        "secret": os.getenv("SECRET"),
                    }
                ),
                "s3": s3.s3_resource.configured(
                    {
                        "bucket_name": os.getenv("BUCKET_NAME"),
                        "region_name": os.getenv("REGION_NAME"),
                        "endpoint_url": os.getenv("ENDPOINT_URL"),
                        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                        "aws_secret_access_key": os.getenv(
                            "AWS_SECRET_ACCESS_KEY"
                        ),
                    }
                ),
            },
        )
    ]

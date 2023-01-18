import boto3

from typing import Union
from dagster import resource, StringSource


class S3ObjectWrapper:
    def __init__(self, **opts: str):
        self.bucket_name = opts.pop("bucket_name")
        self.client = boto3.client("s3", **opts)

    def list(self, prefix: Union[str, None] = None):
        if not prefix:
            return list(
                self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                )
            )
        else:
            return list(
                self.client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=prefix
                )
            )

    def download(self, object_key: str):
        return self.client.get_object(Bucket=self.bucket_name, Key=object_key)


@resource(
    {
        "bucket_name": StringSource,
        "region_name": StringSource,
        "endpoint_url": StringSource,
        "aws_access_key_id": StringSource,
        "aws_secret_access_key": StringSource,
    }
)
def s3_resource(init_context):
    return S3ObjectWrapper(**init_context.resource_config)

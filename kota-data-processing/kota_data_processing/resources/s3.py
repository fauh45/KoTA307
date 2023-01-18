import boto3

from typing import Union
from dagster import resource, StringSource


class S3ObjectWrapper:
    def __init__(self, **opts: str):
        self.bucket_name = opts.pop("bucket_name")
        self.client = boto3.client("s3", **opts)

    def _extract_keys_from_list_objects(self, obj_list):
        return [key["Key"] for key in (obj_list.get("Contents") or [])]

    def list(self, prefix: Union[str, None] = None):
        if not prefix:
            return self._extract_keys_from_list_objects(
                self.client.list_objects_v2(
                    Bucket=self.bucket_name,
                )
            )
        else:
            return self._extract_keys_from_list_objects(
                self.client.list_objects_v2(
                    Bucket=self.bucket_name, Prefix=prefix
                )
            )

    def download(self, object_key: str):
        return self.client.get_object(Bucket=self.bucket_name, Key=object_key)

    def upload(self, file_key: str, upload_body: str):
        self.client.put_object(
            Bucket=self.bucket_name, Key=file_key, Body=upload_body
        )


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

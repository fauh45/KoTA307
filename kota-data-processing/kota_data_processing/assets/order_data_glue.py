from dagster import asset

from kota_data_processing.resources.s3 import S3ObjectWrapper


@asset(
    required_resource_keys={"s3"},
    description="List of raw files exported from Shopify Shop orders",
)
def raw_exported_data(context):
    s3obj: S3ObjectWrapper = context.resources.s3
    exported_obj_list = s3obj.list(prefix="Exported/")

    return exported_obj_list

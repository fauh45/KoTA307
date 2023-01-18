import shopify

from dagster import resource, StringSource


@resource(
    {
        "shop_url": StringSource,
        "access_token": StringSource,
        "api_version": StringSource,
    }
)
def shopify_resources(init_context):
    session = shopify.Session(
        init_context.resource_config["shop_url"],
        init_context.resource_config["api_version"],
        init_context.resource_config["access_token"],
    )

    shopify.ShopifyResource.activate_session(session)

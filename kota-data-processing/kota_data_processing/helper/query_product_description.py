import json
import shopify

cache = dict()

def query_product_description(sku: str):
    if sku in cache:
        return cache[sku]

    graphql_query = (
        'query { products(first:1, query:"sku:%s") { edges { node { description } } } }'
        % sku
    )
    print(graphql_query)

    query_result = json.loads(shopify.GraphQL().execute(graphql_query))

    print(query_result)

    edges = query_result["data"]["products"]["edges"]
    description = ""
    if len(edges) < 1:
        return description

    description = edges[0]["node"]["description"]

    cache[sku] = description
    return description


def invalidate_cache():
    global cache

    cache = dict()
from pymilvus import MilvusClient

from app import config
from app.service.embedding.jina_embedding import JinaEmbeddings
output_fields = ["chunk_text", "file_name","parent_id"]
embedding = JinaEmbeddings()
client= MilvusClient(uri=config.milvus_host_url)

query = "Individuals Onboarding在哪一个界面的哪一个位置"
client.use_database("test_database")
result_by_search = client.search(
            collection_name="TestInfo",
            data=[embedding.embed_query(query)],
            anns_field="dense_vector",
            output_fields=output_fields,
        )


result_by_query = client.query(
            collection_name="TestInfo",
            filter=f"chunk_id == '407431761472131072'",
            output_fields=output_fields
        )

print("search返回的结果:")
print(result_by_search[0]) # [list[dict]- 需要“entity”进行提取


print("search返回的结果:")
print(result_by_query[0]) # list[dict] -直接取chunk_text
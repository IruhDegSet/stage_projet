import argparse
from utils_query import query_rag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("file_type", type=str, choices=["txt", "csv"], help="The type of query file (txt or csv).")
    args = parser.parse_args()
    query_text = args.query_text
    file_type = args.file_type
    response_text = query_rag(query_text, file_type)
    print(response_text)

if __name__ == "__main__":
    main()

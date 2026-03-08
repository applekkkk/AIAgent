import yaml

from utils.path_handler import get_abs_path


def load_config(kind: str, encoding: str = 'utf-8'):
    with open(get_abs_path(f'config/{kind}.yml'), encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


rag_conf = load_config('rag')
chroma_conf = load_config('chroma')
prompts_conf = load_config('prompts')
agent_conf = load_config('agent')

if __name__ == '__main__':
    print(rag_conf['chat_model_name'])

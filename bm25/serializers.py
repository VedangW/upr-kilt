from utils import omit_ends

class Serializer:
    def __init__(self):
        self.pid2title = {}

    def serialize(self):
        pass


class TsvSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def serialize(self, split):
        pid = split[0].strip()
        wiki_id = split[1]
        passage_id = split[2]
        text = omit_ends(split[3].strip(), '"')  # Stripping unnec. quotes
        title = omit_ends(split[4].strip(), '"')
        start_span = split[5]
        end_span = split[6]
        self.pid2title[pid] = title  # Must store title to extract text after search
        return { 
            'id': pid, 
            'wikipedia_id': wiki_id, 
            'passage_id': passage_id, 
            'start_span': start_span, 
            'end_span': end_span, 
            'contents': f'{title} {text}'
        }


class UprSerializer(Serializer):
    def __init__(self):
        super().__init__()

    def serialize(self, split):
        pid = split[0].strip()
        text = omit_ends(split[1].strip(), '"')  # Stripping unnec. quotes
        title = omit_ends(split[2].strip(), '"')
        self.pid2title[pid] = title  # Must store title to extract text after search
        return { 
            'id': pid,
            'contents': f'{title} {text}'
        }
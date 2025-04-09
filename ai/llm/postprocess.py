def get_csv_from_llm(res):
    if '<answer>' in res:
        start = res.find('<answer>')
        if start != -1:
            res = res[start+len('<answer>'):]
        end = res.find('</answer>')
        if end != -1:
            res = res[:end]

    res = res.strip()
    res = res.replace('\n\n', '\n')
    start_tag, end_tag = '```csv', '```'
    left = res.rfind(start_tag)
    if left == -1:
        start_tag = '```'
        left = res.rfind('```')
    right = res.rfind(end_tag)
    if left != -1 and right != -1 and right > left:
        res = res[left+len(start_tag):right].strip()
    else:
        res = res.strip('`')

    return res


def get_list_from_llm(res):
    if '<answer>' in res:
        start = res.find('<answer>')
        if start != -1:
            res = res[start+len('<answer>'):]
        end = res.find('</answer>')
        if end != -1:
            res = res[:end]

    res = res.strip()
    res = res.replace('\n\n', '\n')
    res = res.replace('"，', '",')

    if '"' not in res and "'" not in res:
        res = res.strip('[]')
        res = res.replace('；', ';')
        tmp = res.split(';')
        res = '[' + ','.join([f'"{t}"' for t in tmp]) + ']'

    start_tag, end_tag = '```python', '```'
    left = res.find(start_tag)
    if left == -1:
        start_tag = '```'
        left = res.find('```')
    right = res.rfind(end_tag)
    if left != -1 and right != -1 and right > left:
        res = res[left+len(start_tag):right].strip()
    elif left != -1:
        res = res[left+len(start_tag):].strip()
    else:
        res = res.strip('`')

    left = res.find('[')
    if left != -1:
        res = res[left:]
    else:
        res = '[' + res
    
    if not res.startswith('['):
        res = '[' + res
    if not res.endswith(']'):
        if res.endswith(','):
            res = res.strip(',')
        if res.endswith(',"'):
            res = res[:-2]
        if not res.endswith('"'):
            res += '"'
        res += ']'
    try:
        data = eval(res)
    except:
        print('Error: get_list_from_llm')
        print('res:', res)
        data = []

    return data


def get_json_from_llm(res):
    if '<answer>' in res:
        start = res.find('<answer>')
        if start != -1:
            res = res[start+len('<answer>'):]
        end = res.find('</answer>')
        if end != -1:
            res = res[:end]

    res = res.strip()
    res = res.replace('\\\"', '"').replace('\\n', '\n')
    res = res.replace('\n\n', '\n').replace('\n', '').replace(' ', '')
    res = res.replace('"，', '",').replace('""', '","')
    res = res.replace('}{', '},{')

    start_tag, end_tag = '```json', '```'
    left = res.find(start_tag)
    if left == -1:
        start_tag = '```'
        left = res.find(start_tag)
    right = res.rfind(end_tag)
    if left != -1 and right != -1 and right > left:
        res = res[left+len(start_tag):right].strip()
    elif left != -1:
        res = res[left+len(start_tag):].strip()
    else:
        res = res.strip('`')

    if '``````json' in res:
        tmp = res.split('``````json')
        res = '[' + ','.join(tmp) + ']'
    
    res = res.replace('"""', '"')

    try:
        data = eval(res)
    except:
        print('Error: get_json_from_llm')
        print('res:', res)
        data = {}

    return data

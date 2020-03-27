
def get_bpe_token_positions(clause, context, token_locations):
    """ get the mask for related BPE tokens for one clause (used as the COMET input)
    Args:
        clause(string): output from CoreNLP. In a form of "#sen, #start_index, token-loc, ..."
        context(list): context in raw text form.
        token_locations(list): context tokenized by CoreNLP. In a form of "token-loc"
    Returns:
        sentence_id(int): indicate which utternace is the clause from.
        mask_post_bpe(list): a bpe-token-level mask. 1 for related to the clause, 0 for not.
    """

    def get_token_text(token_loc):
        tokens = token_loc.split('-')
        text, loc = tokens[:-1], tokens[-1]
        if len(tokens) >= 2:
            text = "".join(text)
        if text == "":
            text = ","
        return text

    tokens = clause.split()
    sentence_id, start_token_id, relation = int(tokens[0]), int(tokens[1]), tokens[2:]
    utterance = context[sentence_id]

    tokenized_utterance = token_locations[(sentence_id - 1) // 2].split()

    tokenized_pre_bpe = re.findall(tokenizer.pat, utterance)
    tokenized_pre_bpe_striped = [t.strip() for t in tokenized_pre_bpe]
    mask_pre_bpe = [0] * len(tokenized_pre_bpe)

    dq = deque()
    for t in relation:
        index = tokenized_utterance.index(t)
        token_left = "" if index == 0 else tokenized_utterance[index - 1]
        token_right = "" if index == (len(tokenized_utterance) - 1) else tokenized_utterance[index + 1]
        token_mid = tokenized_utterance[index]
        dq.append((token_left, token_mid, token_right))

    while dq:
        token_left, token_mid, token_right = dq.popleft()
        token_left = get_token_text(token_left)
        token_mid = get_token_text(token_mid)
        token_right = get_token_text(token_right)
        for i, t in enumerate(tokenized_pre_bpe_striped):
            if token_mid == t:
                if ((i == 0) or (token_left == tokenized_pre_bpe_striped[i - 1])) \
                        and (
                        (i == len(tokenized_pre_bpe_striped) - 1) or (token_right == tokenized_pre_bpe_striped[i + 1])):
                    mask_pre_bpe[i] = 1

    # build post bpe mask
    mask_post_bpe = []
    tokenized_pre_bpe = ["".join(tokenizer.byte_encoder[b] for b in token.encode("utf-8")) for token in
                         tokenized_pre_bpe]
    for t, m in zip(tokenized_pre_bpe, mask_pre_bpe):
        mask_post_bpe.extend(m for bpe_token in tokenizer.bpe(t).split(" "))

    return sentence_id, mask_post_bpe

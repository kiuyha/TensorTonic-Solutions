def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    step = chunk_size - overlap
    return [
        tokens[i:i+chunk_size]
        
        for i in range(0, len(tokens), step)
        if i == 0 or i+chunk_size <= len(tokens)
    ]
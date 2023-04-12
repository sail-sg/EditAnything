from maskrcnn_benchmark.modeling.language_backbone import build_tokenizer

if __name__ == '__main__':

    tokenizer2 = build_tokenizer("clip")
    tokenized2 = tokenizer2(
        ["Detectest : fishid. jellyfishioasod. penguinasd. puffin.asd shark. starfish. round stingray"])
    print(tokenized2)

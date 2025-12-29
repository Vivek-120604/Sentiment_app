from transformers import pipeline

def main():
    # force PyTorch backend to avoid TF/Keras 3 incompatibilities
    clf = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english', framework='pt')
    samples = [
        "This movie was fantastic! The acting was great and I loved the plot.",
        "Terrible movie. I wasted two hours and the story made no sense.",
        "It was okay — some parts were enjoyable, others were boring.",
    ]
    for s in samples:
        out = clf(s)
        print(s)
        print(out)
        print('---')

if __name__ == '__main__':
    main()

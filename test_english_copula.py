from parser import parse, ReadoutMethod

def test_english_copula():
    test_cases = [
        "dogs are cats",
        "dogs are bad",
        "dogs are bad cats"
    ]
    
    for sentence in test_cases:
        print(f"\nParsing: {sentence}")
        # Using FIBER_READOUT as it's the default and most robust method in this codebase
        parse(sentence, language="English", verbose=False, readout_method=ReadoutMethod.FIBER_READOUT)
        print("-" * 30)

if __name__ == "__main__":
    test_english_copula()

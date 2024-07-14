from models import Model
import pickle
import argparse


def main(
    model_name, max_new_tokens, temperature, sequences_amount, batch_size, language
):
    with open("messages_Dutch.pkl", "rb") as f:
        messages = pickle.load(f)

    model = Model(model_name, max_new_tokens, temperature, sequences_amount, batch_size)
    input_ids = model.prepare_input(messages)
    results = model.generate(input_ids)

    file_name = f"out/results_{model_name}_{language}.pkl"
    with open(file_name, "wb") as f:
        pickle.dump(results, f)

    # return results


if __name__ == "__main__":

    # add args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="LLama",
        type=str,
        help="choose from LLama, Aya and Mixtral",
    )
    parser.add_argument(
        "--max_new_tokens", default=512, type=int, help="maximum new tokens"
    )
    parser.add_argument("--temperature", default=0, type=float, help="temperature")
    parser.add_argument(
        "--sequences_amount", default=1, type=int, help="sequences_amount"
    )
    parser.add_argument("--batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--language", default="English", type=str, help="Language")

    args = parser.parse_args()
    kwargs = vars(args)

    main(**kwargs)

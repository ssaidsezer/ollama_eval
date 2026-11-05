import json
from editdistance import distance

# Load JSON data
with open('ATBK_Pipeline/reports/whisper_benchmark.json') as f:
    data = json.load(f)

print(f"\n" + "-"*80 + "\n")

for item in data:
    model_name = item["model"]
    transcriptions = item["transcriptions"]
    total_insertions = 0
    total_deletions = 0
    total_substitutions = 0
    total_distance = 0


    for transcription in transcriptions:
        # Extract output and actual texts
        actual_text = transcription['original']
        output_text = transcription['transcribed']

        # Split texts into words
        output_words = output_text.split()
        actual_words = actual_text.split()

        #remove all punctuation from output_words and actual_words
        output_words = [word.strip('.,!?;:"()') for word in output_words
                        if word.strip('.,!?;:"()')]
        actual_words = [word.strip('.,!?;:"()') for word in actual_words
                        if word.strip('.,!?;:"()')]

        # Calculate Levenshtein distance
        d = distance(output_words, actual_words)
        total_distance += d

        # Count substitutions, insertions, and deletions
        substitutions = 0
        insertions = 0
        deletions = 0
        for i in range(min(len(output_words), len(actual_words))):
            if output_words[i] != actual_words[i]:
                substitutions += 1
        if len(output_words) > len(actual_words):
            insertions = len(output_words) - len(actual_words)
        elif len(output_words) < len(actual_words):
            deletions = len(actual_words) - len(output_words)

        # Calculate WEr (word error)
        total_insertions += insertions
        total_deletions += deletions
        total_substitutions += substitutions

    total_word_count = 0
    total_word_count = sum(len(t['original'].split()) for t in transcriptions)
    total_generated_word_count = sum(len(t['transcribed'].split()) for t in transcriptions)
    avg_wer = ((total_insertions + total_deletions + total_substitutions) / total_word_count) * 100
    avg_distance = total_distance / total_word_count

    total_time = sum(t['time'] for t in transcriptions)
    avg_time_per_word = total_time / total_word_count

    print(f'Model: {model_name}\n Average WER for a word: {avg_wer:.2f}%\n Average Levenshtein Distance per word: {avg_distance:.2f}\n Average Time per Word: {avg_time_per_word:.5f} seconds\n')
    print(f"total_word_count: {total_word_count}")
    print(f"total_generated_word_count: {total_generated_word_count}")
    print(f"number of transcriptions: {len(transcriptions)}")
    print(f"total_insertions: {total_insertions}")
    print(f"total_deletions: {total_deletions}")
    print(f"total_substitutions: {total_substitutions}")
    print(f"\n" + "-"*80 + "\n")

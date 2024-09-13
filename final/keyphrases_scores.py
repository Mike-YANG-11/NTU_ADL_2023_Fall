import json


# List of book tags
book_genre = [
    "Fiction",
    "Non-fiction",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Fantasy",
    "Historical Fiction",
    "Biography",
    "Autobiography",
    "Self-Help",
    "Thriller",
    "Horror",
    "Poetry",
    "Adventure",
    "Travel",
    "Science",
    "Philosophy",
    "Psychology",
    "Business",
    "Economics",
    "Politics",
    "Memoir",
    "Young Adult",
    "Children's Literature",
    "Classic",
    "Crime",
    "Comedy",
    "Drama",
    "Suspense",
    "Supernatural",
    "Cultural",
    "Inspirational",
    "Art",
    "Music",
    "Cooking",
    "Gardening",
    "Parenting",
    "Education",
    "Fitness",
    "Technology",
    "Programming",
    "History",
    "Environmental",
    "Mythology",
    "Religion",
    "Sports",
    "Wellness",
    "Sociology",
    "Linguistics",
    "True Crime",
]

dicts = json.load(open("./output_distilbert_v2.json", "r"))
keyphrases_dicts = []
for i in range(len(dicts)):
    keyphrases_dict = {}
    keyphrases_dict["keyphrases"] = dicts[i]["keyphrases"]
    keyphrases_dicts.append(keyphrases_dict)

# convert book tags to lowercase string
book_genre_str = str(book_genre).lower()
# check if keyphrase is in book tags
genre_related_tags = {}
other_tags = {}
for i in range(len(keyphrases_dicts)):
    for keyphrase, score in keyphrases_dicts[i]["keyphrases"].items():
        for word in keyphrase.split(" "):
            if word in book_genre_str:
                genre_related_tags[keyphrase] = score
                break
        other_tags[keyphrase] = score

# sort the keyphrases by score (from low to high)
genre_related_tags = sorted(genre_related_tags.items(), key=lambda x: x[1])
other_tags = sorted(other_tags.items(), key=lambda x: x[1])
genre_related_tags = [{k: v} for k, v in genre_related_tags]
other_tags = [{k: v} for k, v in other_tags]
# rescale the score so the lowest score is 10, the second lowest score is 11, and so on
for i in range(len(genre_related_tags)):
    tag, score = list(genre_related_tags[i].items())[0]
    genre_related_tags[i] = {tag: (10 + i) * 2}
for i in range(len(other_tags)):
    tag, score = list(other_tags[i].items())[0]
    other_tags[i] = {tag: int(score * 2)}
# sort the keyphrases by score (from high to low)
genre_related_tags = sorted(genre_related_tags, key=lambda x: float(list(x.values())[0]), reverse=True)
other_tags = sorted(other_tags, key=lambda x: float(list(x.values())[0]), reverse=True)
# merge the two lists
merged_tags = genre_related_tags + other_tags
# delete repeated keyphrases
result = {}
for i in range(len(merged_tags)):
    tag, score = list(merged_tags[i].items())[0]
    if tag.title() not in result:
        result[tag.title()] = score
print(result)
# save the result to a json file
with open("tags_with_scores.json", "w") as outfile:
    json.dump(result, outfile)

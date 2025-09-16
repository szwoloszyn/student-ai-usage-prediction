import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
DATASET = "dataset/ai_assistant_usage_student_life.csv"

def load_ai_usage_data():
    if not Path(DATASET).is_file():
        raise TypeError
    return pd.read_csv(Path("dataset/ai_assistant_usage_student_life.csv"))

data = load_ai_usage_data()
print(data['SessionLengthMin'])
print(data.head())

## dropping SessionID and cutting SessionDate to year only
X = data.drop(columns=["UsedAgain", "SessionID"])
y = data["UsedAgain"].astype(int)  # True/False -> 1/0

### percentage of students returning to AI depending on their SatisfactionRating
used_again_pct = (
    data.groupby('AI_AssistanceLevel')['UsedAgain']
      .mean() * 100
)
plt.plot(used_again_pct)

print(used_again_pct)

### sessionLength depending on SatisfactionRating. Blue dots = returned, Red = did not
# plt.plot(data['SatisfactionRating'], data['SessionLengthMin'], 'o')
# plt.show()
df_true = data[data['UsedAgain'] == True]
df_false = data[data['UsedAgain'] == False]

### Plot them with different colors
# plt.plot(df_true['SatisfactionRating'], df_true['SessionLengthMin'], 'o', color='blue', label='Used Again')
# plt.plot(df_false['SatisfactionRating'], df_false['SessionLengthMin'], 'o', color='red', label='Not Used Again')

# plt.xlabel('Satisfaction Rating')
# plt.ylabel('Session Length (min)')
# plt.title('Session Length vs Satisfaction Rating')
# plt.legend()
# plt.show()

# ## Histogram of students' overall Satisfaction rating
# plt.hist(data['SatisfactionRating'], alpha=0.5, label='All Students')

# ## Histogram of only those who returned to AI
# plt.hist(data[data['UsedAgain'] == True]['AI_AssistanceLevel'], 
#        alpha=0.5, label='Used Again', color='green')
# plt.hist(data[data['UsedAgain'] == False]['AI_AssistanceLevel'], 
#        alpha=1, label='Used Again', color='red')
# plt.legend()

# Group by satisfaction rating
percentages = (
    data.groupby('AI_AssistanceLevel')["UsedAgain"]
        .mean() * 100   # mean of True/False is % True
)

# Plot as bar chart
percentages.plot(kind="bar")

plt.ylabel("% Used Again")
plt.xlabel("Satisfaction Rating")
plt.title("Percentage of 'Used Again' by Satisfaction Rating")
plt.show()

### randomly sample 100 entries

sampled_data = data.sample(n=100, random_state=42)

# sample_true = sampled_data[sampled_data['UsedAgain'] == True]
# sample_false = sampled_data[sampled_data['UsedAgain'] == False]
# plt.plot(sample_true['SatisfactionRating'], sample_true['TotalPrompts'], 'o', color='blue', label='Used Again')
# plt.plot(sample_false['SatisfactionRating'], sample_false['TotalPrompts'], 'o', color='red', label='Not Used Again', alpha = 0.6)
# print(sampled_data)

# sns.lmplot(
#     data=sampled_data,
#     x="AI_AssistanceLevel",
#     y="TotalPrompts",
#     col="Discipline",
#       hue="UsedAgain",
#     palette={True: "blue", False: "red"},
#     fit_reg=False
# )
print("AAA")
data.groupby('UsedAgain')['SessionLengthMin'].describe()

#plt.plot(sampled_data['SatisfactionRating'], sampled_data['SessionLengthMin'], 'o')
plt.show()

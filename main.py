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

### percentage of students returning to AI depending on their SatisfactionRating
used_again_pct = (
    data.groupby('SatisfactionRating')['UsedAgain']
      .mean() * 100
)
# plt.plot(used_again_pct)

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

### Histogram of students' overall Satisfaction rating
#plt.hist(data['SatisfactionRating'], alpha=0.5, label='All Students')

### Histogram of only those who returned to AI
#plt.hist(data[data['UsedAgain'] == True]['SatisfactionRating'], 
#        alpha=0.5, label='Used Again')

### randomly sample 100 entries

sampled_data = data.sample(n=1000, random_state=42)

sample_true = sampled_data[sampled_data['UsedAgain'] == True]
sample_false = sampled_data[sampled_data['UsedAgain'] == False]
plt.plot(sample_true['SatisfactionRating'], sample_true['TotalPrompts'], 'o', color='blue', label='Used Again')
plt.plot(sample_false['SatisfactionRating'], sample_false['TotalPrompts'], 'o', color='red', label='Not Used Again')
print(sampled_data)

sns.lmplot(
    data=sampled_data,
    x="SatisfactionRating",
    y="SessionLengthMin",
    col="Discipline",
      hue="UsedAgain",
    palette={True: "blue", False: "red"},
    fit_reg=False
)

#plt.plot(sampled_data['SatisfactionRating'], sampled_data['SessionLengthMin'], 'o')
plt.show()

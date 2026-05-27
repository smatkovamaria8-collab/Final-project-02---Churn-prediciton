import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

data = pd.read_csv("internet_service_churn.csv")
print(data.head(10))
print(data.info())

print(data["subscription_age"].describe())
print(data["churn"].describe())

sns.countplot(x="is_tv_subscriber", hue="is_tv_subscriber", data=data, legend= True)
plt.xlabel("Чи підписаний на телебачення", fontsize="medium", color="midnightblue")
plt.ylabel("Кількість", fontsize="medium", color="midnightblue")
plt.legend(["0 - без підписки", "1 - з підпискою"], fontsize=10)
plt.show()

# sns.countplot(x="is_movie_package_subscriber", data=data)
# plt.show()

# sns.lineplot(data=df)


# sns.countplot(x="churn", data=data)
# plt.show()
import psycopg2
import pandas as pd

connection = psycopg2.connect(
    database="database",
    user="user",
    password="password",
    host="ip address")

cursor = connection.cursor()

# db structure:
# article_metadata(article, id, title, content)
# article_metadata(article_id, key, value)

cursor.execute("""SELECT DISTINCT key, value FROM article_metadata order by key, value""")

result = [(key, value) for key, value in cursor.fetchall()]

fields = [key for key, value in result]

classes=[f"{key}={value}" for key, value in result]

print(classes)

cursor = connection.cursor()
cursor.execute("""SELECT DISTINCT id, concat(title, ' ', content) as title, key, value FROM article left join article_metadata on id=article_id order by id""")

x = pd.DataFrame({"title": []})
y = pd.DataFrame({classname : [] for classname in classes})
y_fields = pd.DataFrame({field : [] for field in fields})

index = 0
last_id = None
for id, title, key, value in cursor.fetchall():
    if id != last_id:
        last_id=id
        index+=1
        # set all classes to zero
        for classname in classes:
            y.at[index, classname] = 0
        # set all fields to '-'
        for field in fields:
            y_fields.at[index, field] = "-"
    x.at[index, "title"] = title
    y.at[index, f"{key}={value}"] = 1
    y_fields.at[index, key] = value


x.to_csv("data/x.csv", index=False)
y.to_csv("data/y.csv", index=False)
y_fields.to_csv("data/y_fields.csv", index=False)

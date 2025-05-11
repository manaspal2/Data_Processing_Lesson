import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
from termcolor import colored, cprint

print (colored ("##################################","red", attrs=['bold']))
print (colored ("## Connect to database          ##","red", attrs=['bold']))
print (colored ("##################################","red", attrs=['bold']))
conn = psycopg2.connect(
    host="localhost",
    database="postgres",
    user="postgres",
    password="password",
    port=5432
)
cursor = conn.cursor()

engine = create_engine('postgresql://postgres:password@localhost:5432/postgres')

print (colored ("##################################","red", attrs=['bold']))
print (colored ("## Read the dataset from csv    ##","red", attrs=['bold']))
print (colored ("##################################","red", attrs=['bold']))
monarchs_df = pd.read_csv("leaders/monarchs.csv")
print (monarchs_df.head())
print (colored ("##################################","red", attrs=['bold']))
presidents_df = pd.read_csv("leaders/presidents.csv")
print (presidents_df.head())
print (colored ("##################################","red", attrs=['bold']))
prime_minister_terms_df = pd.read_csv("leaders/prime_minister_terms.csv")
print (prime_minister_terms_df.head())
print (colored ("##################################","red", attrs=['bold']))
prime_ministers_df = pd.read_csv("leaders/prime_ministers.csv")
print (prime_ministers_df.head())
print (colored ("##################################","red", attrs=['bold']))
states_df = pd.read_csv("leaders/states.csv")
print (states_df.head())

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## drop table if it already exists ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
cursor.execute('drop table if exists monarchs')
conn.commit()
cursor.execute('drop table if exists presidents')
conn.commit()
cursor.execute('drop table if exists prime_minister_terms')
conn.commit()
cursor.execute('drop table if exists prime_ministers')
conn.commit()
cursor.execute('drop table if exists states')
conn.commit()

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a monarchs table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE monarchs(country varchar(50), continent varchar(50) ,
monarch varchar(50));'''
cursor.execute(sql)
conn.commit()
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
monarchs_df.to_sql('monarchs', con=engine, if_exists= 'replace', index=False)
print (colored ("########################################################","red", attrs=['bold']))
print (colored ("## Verify the data of monarchs table in database       ##","red", attrs=['bold']))
print (colored ("#########################################################","red", attrs=['bold']))
sql = '''SELECT * FROM monarchs;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a presidents table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE presidents(country varchar(50), continent varchar(50) ,
president varchar(50));'''
cursor.execute(sql)
conn.commit()
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
presidents_df.to_sql('presidents', con=engine, if_exists= 'replace', index=False)
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Verify the data in database       ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
sql = '''SELECT * FROM presidents;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a prime_minister_terms table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE prime_minister_terms(prime_minister varchar(50), pm_start int);'''
cursor.execute(sql)
conn.commit()
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
prime_minister_terms_df.to_sql('prime_minister_terms', con=engine, if_exists= 'replace', index=False)
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Verify the data in database       ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
sql = '''SELECT * FROM prime_minister_terms;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a prime_ministers table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE prime_ministers(country varchar(50), continent varchar(50),
prime_minister varchar(50));'''
cursor.execute(sql)
conn.commit()
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
prime_ministers_df.to_sql('prime_ministers', con=engine, if_exists= 'replace', index=False)
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Verify the data in database       ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
sql = '''SELECT * FROM prime_ministers;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a states table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE states(country varchar(50), continent varchar(50),
indep_year varchar(50));'''
cursor.execute(sql)
conn.commit()
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
states_df.to_sql('states', con=engine, if_exists= 'replace', index=False)
print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Verify the data in database       ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
sql = '''SELECT * FROM states;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("##############################################################################","red", attrs=['bold']))
print (colored ("## Inner Join to find how many countries have prime minister and president  ##","red", attrs=['bold']))
print (colored ("##############################################################################","red", attrs=['bold']))
sql = '''SELECT prime_ministers.country, prime_ministers.continent, prime_minister, president
FROM presidents
INNER JOIN prime_ministers
ON presidents.country = prime_ministers.country;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 
print (colored ("#################### SAME INNER JOIN DIFF FORMAT ###########################","red", attrs=['bold']))
sql = '''SELECT presidents.country, presidents.continent, prime_minister, president
FROM presidents
INNER JOIN prime_ministers
USING(country);'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 
print (colored ("#################### SAME INNER JOIN DIFF FORMAT ###########################","red", attrs=['bold']))
sql = '''SELECT p2.country, p2.continent, prime_minister, president
FROM prime_ministers AS p1
INNER JOIN presidents AS p2
ON p2.country = p1.country;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("########################","cyan", attrs=['bold']))
print (colored ("## LEFT JOIN EXAMPLE ##","cyan", attrs=['bold']))
print (colored ("########################","cyan", attrs=['bold']))
sql = '''SELECT *
FROM prime_ministers AS p1
LEFT JOIN presidents AS p2
ON p2.country = p1.country;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("########################","yellow", attrs=['bold']))
print (colored ("## RIGHT JOIN EXAMPLE ##","yellow", attrs=['bold']))
print (colored ("########################","yellow", attrs=['bold']))
sql = '''SELECT *
FROM prime_ministers AS p1
RIGHT JOIN presidents AS p2
ON p2.country = p1.country;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("##################################","red", attrs=['bold']))
print (colored ("## Read the dataset from csv    ##","red", attrs=['bold']))
print (colored ("##################################","red", attrs=['bold']))
cities_df = pd.read_csv("countries/cities.csv")
print (cities_df.head())
cursor.execute('drop table if exists cities')
conn.commit()

print (colored ("#####################################","red", attrs=['bold']))
print (colored ("## Create a monarchs table         ##","red", attrs=['bold']))
print (colored ("#####################################","red", attrs=['bold']))
sql = '''CREATE TABLE cities(name varchar(50), country_code varchar(50),
city_proper_pop real,
metroarea_pop real, 
urbanarea_pop real);'''
cursor.execute(sql)
conn.commit()

print (colored ("#######################################","red", attrs=['bold']))
print (colored ("## Add whole dataframe into database ##","red", attrs=['bold']))
print (colored ("#######################################","red", attrs=['bold']))
cities_df.to_sql('cities', con=engine, if_exists= 'replace', index=False)
sql = '''SELECT * FROM cities;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i) 

print (colored ("########################","red", attrs=['bold']))
print (colored ("## INNER JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT * 
FROM cities
INNER JOIN countries
ON cities.country_code = countries.code;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

print (colored ("########################","red", attrs=['bold']))
print (colored ("## INNER JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT cities.name AS city, countries.name AS country, region
FROM cities
INNER JOIN countries
ON cities.country_code = countries.code;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

print (colored ("########################","red", attrs=['bold']))
print (colored ("## INNER JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT cities.country_code AS code, cities.name AS city, countries.name AS country, region
FROM countries
INNER JOIN cities
ON countries.code = cities.country_code;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    if 'IND' in i:
        print (i)  

print (colored ("########################","red", attrs=['bold']))
print (colored ("## INNER JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT languages.code AS code, languages.name AS lang_name, countries.name AS country
FROM languages
INNER JOIN countries
ON languages.code = countries.code;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    if 'Hindi' in i:
        print (i)  
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT c.name AS country, l.name AS language
FROM countries AS c
INNER JOIN languages AS l
USING(code)
WHERE l.name = 'Bhojpuri'
LIMIT 10;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

print (colored ("########################","red", attrs=['bold']))
print (colored ("## MULTI JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT p1.country, p1.prime_minister AS pm, p2.president AS president, pmt.pm_start AS start
FROM prime_ministers AS p1
INNER JOIN presidents AS p2
USING(country)
INNER JOIN prime_minister_terms AS pmt
USING(prime_minister);'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

print (colored ("########################","red", attrs=['bold']))
print (colored ("## MULTI JOIN EXAMPLE ##","red", attrs=['bold']))
print (colored ("########################","red", attrs=['bold']))
sql = '''SELECT name, p.year, fertility_rate, e.year, unemployment_rate
FROM countries AS c
INNER JOIN populations AS p
ON c.code = p.country_code
INNER JOIN economies AS e
ON c.code = e.code 
AND p.year = e.year;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

print (colored ("########################","green", attrs=['bold']))
print (colored ("## INNER JOIN EXAMPLE ##","green", attrs=['bold']))
print (colored ("########################","green", attrs=['bold']))
sql = '''SELECT c1.name AS city, code, c2.name AS country, region, city_proper_pop
FROM cities AS c1
INNER JOIN countries AS c2
ON c1.country_code = c2.code
ORDER BY code DESC;'''
cursor.execute(sql) 
#for i in cursor.fetchall(): 
#    print (i)  
print ("Length of data record:", len(cursor.fetchall()))
print (colored ("########################","cyan", attrs=['bold']))
print (colored ("## LEFT JOIN EXAMPLE ##","cyan", attrs=['bold']))
print (colored ("########################","cyan", attrs=['bold']))
sql = '''SELECT c1.name AS city, code, c2.name AS country, region, city_proper_pop
FROM cities AS c1
LEFT JOIN countries AS c2
ON c1.country_code = c2.code
ORDER BY code DESC;'''
cursor.execute(sql) 
#for i in cursor.fetchall(): 
#    print (i)  
print ("Length of data record:", len(cursor.fetchall()))

print (colored ("#########################################","cyan", attrs=['bold']))
print (colored ("Complete the LEFT JOIN with the countries table on the \
left and the economies table on the right on the code field.\
Filter the records from the year 2010.\
To calculate per capita GDP per region, begin by grouping by region\
After your GROUP BY, choose region in your SELECT statement, \
followed by average GDP per capita using the AVG() function, with \
AS avg_gdp as your alias. \
Order the result set by the average GDP per capita from highest to lowest.","cyan", attrs=['bold']))
print (colored ("#########################################","cyan", attrs=['bold']))
sql = '''SELECT region, AVG(gdp_percapita) as avg_gdp
FROM countries AS c
LEFT JOIN economies AS e
ON c.code = e.code
WHERE year = 2010
GROUP BY region
ORDER BY avg_gdp DESC
LIMIT 10;'''
cursor.execute(sql) 
for i in cursor.fetchall(): 
    print (i)  

conn.commit() 
cursor.close()
conn.close()

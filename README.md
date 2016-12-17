# Machine Learning Project2 - Recommender Systems
## Manana Lordkipanidze, Aidasadat Mousavifar, Frederike Dümbgen

Install apache spark on ubuntu 16 (server):

for installation:
Follow these [installation](  
https://medium.com/%E0%B8%AD%E0%B8%B0%E0%B9%84%E0%B8%A3%E0%B8%A2%E0%B8%B1%E0%B8%87%E0%B9%84%E0%B8%87/installing-spark-1-6-1-on-ubuntu-16-04-bb6b60a1b74e#.dh9el8igh)
instructions, for installing required packages use wget and copy the download link from the website. 

For jdk-8, the link is broken, but it can be installed via command line using [these](
http://tecadmin.net/install-oracle-java-8-jdk-8-ubuntu-via-ppa/)
instructions. Important: JAVA\_HOME in the following instructions is then 
``` bash
/usr/lib/jvm/java-1.8.0-openjdk-amd64/
```

To use pyspark from anaconda, follow [these](
http://dstil.ghost.io/setting-up-apache-py-spark-with-jupyter-notebook-in-arch-linux/)
instructions, replacing the startup file by 

``` python
# Configure the necessary Spark environment
import os  
import sys

pyspark_submit_args = os.environ.get("PYSPARK_SUBMIT_ARGS","")

if not "pyspark-shell" in pyspark_submit_args :  
  pyspark_submit_args += " pyspark-shell"

os.environ["PYSPARK_SUBMIT_ARGS"]= pyspark_submit_args

#spark_home = os.environ.get("SPARK_HOME", '/opt/apache-spark')  
spark_home = os.environ.get("SPARK_HOME")  
sys.path.insert(0, spark_home + "/python")

# Add the py4j to the path.
# You may need to change the version number to match your install
sys.path.insert(0,os.path.join(spark_home, "python/lib/py4j-0.9-src.zip"))

# Initialize PySpark
major_version = sys.version_info.major  
pyspark_shell_file = os.path.join(spark_home, "python/pyspark/shell.py")

```

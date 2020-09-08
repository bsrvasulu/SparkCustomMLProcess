# SparkCustomMLProcess
 

# Custom ML to Process tif images

**Load Spark library**



```python
%load_ext autoreload
%autoreload 2
```


```python
import findspark
findspark.init()
```


```python
from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark import SparkFiles
```

**Load pattern libaries**


```python
import numpy as np
import pandas as pd
import cv2
import imutils
import math
import os
import time
from datetime import datetime
import traceback
import statistics
from PIL import Image
```

**Initilize Spark and Spark context**
<div>Configure memory</div>



```python
spark = SparkSession \
    .builder \
    .master("yarn") \
    .appName("IMAGE_Process") \
    .config("spark.executor.memory", "20gb") \
    .config("spark.executor.cores", "2") \
    .config("spark.cores.max", "2") \
    .enableHiveSupport()\
    .getOrCreate()
sc = spark.sparkContext
sc
```

### RDD to iterate the directory and collect tif files


```python
hadoop = sc._jvm.org.apache.hadoop
fs = hadoop.fs.FileSystem
conf = hadoop.conf.Configuration()
```


```python
fs = hadoop.fs.FileSystem
conf = hadoop.conf.Configuration() 
def listtifFiles(hdfspath, lft):
    path = hadoop.fs.Path(hdfspath)
    ld = fs.get(conf).listStatus(path)
    for d1 in ld:
        if fs.get(conf).isDirectory(d1.getPath()):
            #print('dir: ' + str(d1.getPath()))
            listtifFiles(str(d1.getPath()), lft)
        else:
            if str(d1.getPath()).endswith('.tif') :
                #print('tif: ' + str(d1.getPath()))
                lft.append(str(d1.getPath()))    
```


```python
fs = hadoop.fs.FileSystem
conf = hadoop.conf.Configuration() 
path = hadoop.fs.Path('hdfs://base directory')
lf = [f.getPath() for f in fs.get(conf).listStatus(path)]
```


```python
lf2 = lf#[50:75]
lftif = []
for d1 in lf2:
    #print(str(d1))
    listtifFiles(str(d1), lftif)
```


```python
len(lftif)
```


```python
lftif[0]
```


```python
tif_file = lftif[0]
tif_file
```

## Custom Process image files


```python
#img_loc = tif_file
#binaryRdd=sc.binaryFiles(img_loc)
#bin_data=binaryRdd.map(read_array)
#img_data = bin_data.take(2)
```

**RDD-1: Load tif files as binary files**


```python
tif_file_path = 'hdfs://vase directory/*/'
binaryRawRdd = sc.binaryFiles(tif_file_path)
```

**Procedure to convert binary to numpy array**


```python
def read_array(rdd):
    np_array = np.frombuffer(rdd[1][:], dtype=np.uint8)
    return rdd[0], np_array
```

**RDD-2: Convert binary data to numpy array**


```python
bin_np_array_data=binaryRawRdd.map(read_array)
```

**Excute RDDs and collect fist two elements**


```python
#process_data = bin_np_array_data.take(2)
#print(process_data[1])
```

**Add required python file and config file**


```python
sc.addFile('config.txt')
```


```python
sc.addPyFile('CustomMLProcess.py')
```


```python
import uuid 
unique_id = uuid.uuid1()
broadcastUUID = sc.broadcast([unique_id])
```

*Add import statement, so that added files is available at worker nodes*


```python
import customMLProcess as cml
```

*Procedure to ML process*


```python
def process_image(rdd):
    image_file = rdd[0]
    print(image_file)
    print(rdd[1])
    split_image_file = image_file.split('/')
    experiment_number = split_image_file[-3]
    disk = split_image_file[-2]
    date_str = split_image_file[-4]
    #create object
    custom_ml_process = cml.customMLProcess()
    #load config file
    config_file = SparkFiles.get('config.txt')
    #process and get process data
    csv_ml_data = custom_ml_process.image_process('Location', image_file, rdd[1], config_file)
    uuid = broadcastUUID.value[0] 
    return (uuid, image_file, experiment_number, disk, csv_ml_data)
```

**RDD-3: Process image and collect ml data**


```python
processed_json_rdd=bin_np_array_data.map(process_image)
```

**Execute Spark lineage and collect defects data** 


```python
processed_json_data = processed_json_rdd.take(2)
```


```python
len(processed_json_data)
```


```python
processed_json_data[0]
```


```python
processed_json_data[0][4][0][0]
```

**Convert JSON data to pandas data frame**


```python
import json
json_defect_data = json.loads(processed_json_data[0][4][0][0])
df_defect = pd.read_json(json_defect_data)
df_defect['EXPERIMENT_NUMBER_2'] = processed_json_data[0][1] 
df_defect.head()
```


```python
df_defect.columns
```


```python
#help(df_defect)
```

## Load pandas dataframe to hive


```python
sdf = spark.createDataFrame(df_defect)
sdf.printSchema()
```

**Save dataframe to new hive table 


```python
sd = spark.sql('show databases').show()
sd = spark.sql('show tables from gold').show()
```


```python
# Save df to a new table in Hive
#df.write.mode("overwrite").saveAsTable("test_db.test_table2")
# Show the results using SELECT
#spark.sql("select * from test_db.test_table2").show()
#df.write.mode("append").saveAsTable("test_db.test_table2")
```


```python

```


```python

```

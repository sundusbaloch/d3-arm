# Running Spark locally on Windows

Spark itself is just a pip install, but on Windows it needs two Hadoop helper
files to write parquet, otherwise you get errors about `winutils.exe` or
`NativeIO$Windows`.

1. **Java 17** - install Eclipse Temurin 17 (JDK). Check with `java -version`.
   Spark 3.5 officially supports Java 8, 11 and 17, so 17 is the safe choice.
2. **Python 3.10, 3.11 or 3.12** (PySpark 3.5 fails to install on 3.13).
3. **winutils** - download `winutils.exe` and `hadoop.dll` built for Hadoop 3.3.x
   (search GitHub for "winutils hadoop 3.3"; the cdarlint/winutils repo is the usual one).
   Put both in `C:\hadoop\bin`. Also copy `hadoop.dll` into `C:\Windows\System32`.
4. Environment variables (System Properties -> Environment Variables):
   - `JAVA_HOME` = the Temurin install folder, e.g. `C:\Program Files\Eclipse Adoptium\jdk-17...`
   - `HADOOP_HOME` = `C:\hadoop`
   - add `%JAVA_HOME%\bin` and `%HADOOP_HOME%\bin` to `Path`
   - `PYSPARK_PYTHON` = `python`
5. New terminal, then in the repo folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   pytest -q tests/
   ```
   If the two tests pass, Spark can read and write parquet and everything else will work.

Memory: in local mode the whole job lives in the driver JVM, so `--mem 8g` is
the Spark memory. On a 16 GB laptop do not go above 10g.

The Spark UI is at http://localhost:4040 while a job is running - screenshots
of the stage times are useful for the scaling section of the report.

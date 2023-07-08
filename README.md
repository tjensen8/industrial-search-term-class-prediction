# Data Science Assignment

This repo contains the data science assignment, pipeline, and results for predicting the class of a particular type of search term used in an industrial manufacturing supplier search.  

## File Structure

___
### 1. Database Creation
The files are organized by use. The initial sqlite database I created to allow for efficient querying and memory management can be found in the `/database` directory. The file that creates and populates the database is located in `/database/create_database.py`.

___

### 2. Analysis
The final EDA and analysis that I conducted on the provided dataset is located in `/src/analysis.ipynb` and the associated functions that I built are located in the similarly named `analysis.py` file in the same directory. This file also contains and runs the pipeline for the model.

___

### 3. Modeling

The final model that I selected with the associated parameters is located in the `/src/modeling.ipynb` file and contains the full approach to the modeling. This file does not have a separate function package because I wanted to fit all of the supporting functions in the same notebook for review ease of use in colab.

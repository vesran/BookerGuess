import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import your data
users = pd.read_csv('./data/train_users_2.csv')
test_users = pd.read_csv('./data/test_users.csv')
sessions = pd.read_csv('./data/sessions.csv')
ages = pd.read_csv('./data/age_gender_bkts.csv')
countries = pd.read_csv('./data/countries.csv')
# ## All users are from US

# How many users in train ? in test ? in sessions ?
users.id.nunique()
test_users.id.nunique()
sessions.user_id.nunique()
# ## There are 213 451 users in train and 62 096 users in test. There are 135 483 user session recorded.
# ## We do not always have information about an user session. Session information are bonus or to be guessed.


# What are the country destinations (target variable) ? Is the target variable balanced in train ?
users.country_destination.hist()
users.groupby('country_destination').count()
# ## The country destinations are 'US', 'FR', 'CA' (Canada), 'GB', 'ES' (Spain), 'IT' (Italy), 'PT'(Portugal),
# ## 'NL' (Netherlands),'DE' (Germany), 'AU'(Australia),
# ## 'NDF' (no destination found), and 'other'.
# ## The data set is not balanced at all. The class NDF is overrepresented.

# What about null values ?
users.isnull().sum()
users.date_first_booking
# ## Very few variables have null values but there are a lot of null values. Almost half of date_first_booking
# ## are missing => gotta remove it completely. Followed by age (1/3) and first_affiliate_tracked (ok).
# ## Gender can be unknown ! And most of users have unknown gender. Should be dropped.
# ## Maybe use mode function is okay.


# What does the variable looks ? (date format, discrete, continuous...)
users.columns
# ## Timestamp_first_active has to be formatted. yyyy/mm/dd/hh/mm/ss
# ## Signup_app : Web+++, Moweb, iOS+, Android -- relevent ? No correlation - same distribution in each country dest
# ## Signup_flow : the page the user came to sign up from - 0 outnumbered - to bucketize


# What is age_gender_bkts file ? -- Stats about age gender and country destination
# What is countries file ? -- info about countries : distance can be interesting and language too
# What is session file ? -- users actions information

# Return full information about a given user (age, gender, session...)

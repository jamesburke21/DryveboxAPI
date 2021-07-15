from model import NLPModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def build_model():
    model = cpModel()

    # filename = os.path.join(
    #     os.path.dirname(__file__), 'chalicelib', 'all/train.tsv')
    with open('../sentiment_data/train.tsv') as f:
        data = pd.read_csv(f, sep='\t')

    # Json flattening function definition
    def flatten_json(y):
        out = {}
        def flatten(x, name =''):
            if type(x) is dict:
                for a in x:
                    flatten(x[a], name + a + '_')
            elif type(x) is list:
                i = 0
                for a in x:
                    flatten(a, name + str(i) + '_')
                    i += 1
            else:
                out[name[:-1]] = x

        flatten(y)
        return out

    def extract(json, col_names):
        row = {}
        df = pd.DataFrame(columns = col_names)
        for doc in json['DocumentTypes']:
            #accessing the different players as documents
            for player in doc['Documents']:
                row = {}
                players = flatten_json(player['Player'])
                row.update(players)
                enviornment = flatten_json(player['Environment'])
                row.update(enviornment)
                #accessing the individual strokes
                for stroke in player['Strokes']:
                    club = stroke['ClubData']['Name']
                    row.update({'Club' : club})
                    measurment = flatten_json(stroke['Measurement'])
                    row.update(measurment)
                    df = df.append(row, ignore_index = True)
        return df
    documents = data['DocumentTypes']
    columns = ['Id', 'Name', 'Email', 'Hcp', 'Gender', 'Nationality', 'Birthday', 'PlayerCategory_Id', 'PlayerCategory_Name', 'PlayerCategory_Amateur', 'Altitude', 'Temperature', 'Location_Latitude', 'Location_Longitude', 'Club', 'Time', 'Kind', 'TeePosition_0', 'TeePosition_1', 'TeePosition_2', 'LaunchDirection', 'BallSpeed', 'LaunchAngle', 'SpinRate', 'MaxHeight', 'Carry', 'Total', 'CarrySide', 'TotalSide', 'LandingAngle', 'HangTime', 'LastData', 'ReducedAccuracy_0', 'BallTrajectory_0_Kind', 'BallTrajectory_0_XFit_0', 'BallTrajectory_0_XFit_1', 'BallTrajectory_0_XFit_2', 'BallTrajectory_0_XFit_3', 'BallTrajectory_0_XFit_4', 'BallTrajectory_0_XFit_5', 'BallTrajectory_0_XFit_6', 'BallTrajectory_0_YFit_0', 'BallTrajectory_0_YFit_1', 'BallTrajectory_0_YFit_2', 'BallTrajectory_0_YFit_3', 'BallTrajectory_0_YFit_4', 'BallTrajectory_0_YFit_5', 'BallTrajectory_0_YFit_6', 'BallTrajectory_0_ZFit_0', 'BallTrajectory_0_ZFit_1', 'BallTrajectory_0_ZFit_2', 'BallTrajectory_0_ZFit_3', 'BallTrajectory_0_ZFit_4', 'BallTrajectory_0_ZFit_5', 'BallTrajectory_0_ZFit_6', 'BallTrajectory_0_SpinRateFit_0', 'BallTrajectory_0_SpinRateFit_1', 'BallTrajectory_0_SpinRateFit_2', 'BallTrajectory_0_SpinRateFit_3', 'BallTrajectory_0_SpinRateFit_4', 'BallTrajectory_0_TimeInterval_0', 'BallTrajectory_0_TimeInterval_1', 'BallTrajectory_0_ValidTimeInterval_0', 'BallTrajectory_0_ValidTimeInterval_1', 'BallTrajectory_0_MeasuredTimeInterval_0', 'BallTrajectory_0_MeasuredTimeInterval_1', 'BallTrajectory_1_Kind', 'BallTrajectory_1_XFit_0', 'BallTrajectory_1_XFit_1', 'BallTrajectory_1_XFit_2', 'BallTrajectory_1_YFit_0', 'BallTrajectory_1_YFit_1', 'BallTrajectory_1_YFit_2', 'BallTrajectory_1_ZFit_0', 'BallTrajectory_1_ZFit_1', 'BallTrajectory_1_ZFit_2', 'BallTrajectory_1_TimeInterval_0', 'BallTrajectory_1_TimeInterval_1', 'BallTrajectory_1_ValidTimeInterval_0', 'BallTrajectory_1_ValidTimeInterval_1', 'BallTrajectory_2_Kind', 'BallTrajectory_2_XFit_0', 'BallTrajectory_2_XFit_1', 'BallTrajectory_2_XFit_2', 'BallTrajectory_2_YFit_0', 'BallTrajectory_2_YFit_1', 'BallTrajectory_2_YFit_2', 'BallTrajectory_2_ZFit_0', 'BallTrajectory_2_ZFit_1', 'BallTrajectory_2_ZFit_2', 'BallTrajectory_2_TimeInterval_0', 'BallTrajectory_2_TimeInterval_1', 'BallTrajectory_2_ValidTimeInterval_0', 'BallTrajectory_2_ValidTimeInterval_1', 'BallTrajectory_3_Kind', 'BallTrajectory_3_XFit_0', 'BallTrajectory_3_XFit_1', 'BallTrajectory_3_YFit_0', 'BallTrajectory_3_ZFit_0', 'BallTrajectory_3_ZFit_1', 'BallTrajectory_3_TimeInterval_0', 'BallTrajectory_3_TimeInterval_1', 'BallTrajectory_3_ValidTimeInterval_0', 'BallTrajectory_3_ValidTimeInterval_1', 'PlayerDexterity', 'DynamicLie', 'ImpactOffset', 'ImpactHeight', 'AttackAngle', 'ClubPath', 'ClubSpeed', 'DynamicLoft', 'FaceAngle', 'FaceToPath', 'SmashFactor', 'SpinAxis', 'SpinLoft', 'SwingDirection', 'SwingPlane', 'SwingRadius', 'DPlaneTilt', 'LowPointDistance', 'ClubTrajectory_0_Kind', 'ClubTrajectory_0_XFit_0', 'ClubTrajectory_0_XFit_1', 'ClubTrajectory_0_XFit_2', 'ClubTrajectory_0_XFit_3', 'ClubTrajectory_0_YFit_0', 'ClubTrajectory_0_YFit_1', 'ClubTrajectory_0_YFit_2', 'ClubTrajectory_0_YFit_3', 'ClubTrajectory_0_ZFit_0', 'ClubTrajectory_0_ZFit_1', 'ClubTrajectory_0_ZFit_2', 'ClubTrajectory_0_ZFit_3', 'ClubTrajectory_0_TimeInterval_0', 'ClubTrajectory_0_TimeInterval_1', 'ClubTrajectory_1_Kind', 'ClubTrajectory_1_XFit_0', 'ClubTrajectory_1_XFit_1', 'ClubTrajectory_1_XFit_2', 'ClubTrajectory_1_XFit_3', 'ClubTrajectory_1_YFit_0', 'ClubTrajectory_1_YFit_1', 'ClubTrajectory_1_YFit_2', 'ClubTrajectory_1_YFit_3', 'ClubTrajectory_1_ZFit_0', 'ClubTrajectory_1_ZFit_1', 'ClubTrajectory_1_ZFit_2', 'ClubTrajectory_1_ZFit_3', 'ClubTrajectory_1_TimeInterval_0', 'ClubTrajectory_1_TimeInterval_1']
    strokes_df = extract(data, columns)

    #Clean/Prep data
    #Removing columns with many missing values
    percent_missing = {}
    length = len(strokes_df.index)
    for column in strokes_df:
        missing = strokes_df[column].isnull().sum()
        per = missing/length
        percent_missing[column] = per
    features = []
    for feature in percent_missing:
        if percent_missing[feature] == 0.0:
            features.append(feature)
    nan_df = strokes_df[features]

    #Removing columns with no variation
    variance = {}
    numeric = nan_df.select_dtypes(include=[np.floating])
    categorical = nan_df.select_dtypes(exclude=[np.floating])
    for feature in numeric:
        var = np.var(numeric[feature])
        variance[feature] = var
    varFeat = []
    for feature in variance:
        if variance[feature] != 0.0:
            varFeat.append(feature)

    #Adding back categorical variables
    df = pd.concat([categorical, nan_df[varFeat]], axis=1)

    model.vectorizer_fit(df.loc[:, 'Phrase'])
    print('Vectorizer fit complete')

    X = model.vectorizer_transform(df.loc[:, 'Phrase'])
    print('Vectorizer transform complete')
    y = df.loc[:, 'Binary']

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model.train(X_train, y_train)
    print('Model training complete')

    model.pickle_clf()
    model.pickle_vectorizer()

    model.plot_roc(X_test, y_test)


if __name__ == "__main__":
    build_model()

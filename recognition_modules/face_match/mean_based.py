import face_recognition
from pandas import DataFrame



def get_name_mean(encoding,data,tolerance = 0.5):
    # get distances from encoding
    distance = face_recognition.face_distance(encoding,data['encodings'])
    # get names
    nomes = data['names']
    # build data frame
    temp_dict = DataFrame({'dist':distance,
                'names':nomes})
    # grouping and summarize
    df_agg_mean_dist = temp_dict.groupby(['names'])['dist'].agg(['mean'])
    df_agg_mean_dist = df_agg_mean_dist.reset_index()
    # get the min dist
    minimo = df_agg_mean_dist['mean'].min()
    row_min = df_agg_mean_dist[df_agg_mean_dist['mean'] == minimo]
    min_dist = row_min.mean().to_list()[0]
    name = row_min.names.to_list()[0]
    # give name
    if min_dist < tolerance:
        return name
    else:
        return 'Unknown'
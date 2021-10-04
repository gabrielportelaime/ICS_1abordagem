import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


class ICS:

    def __init__(self, laplace_smoothing=0.01, omega=0.5):
        self.__users = pd.read_csv("1_social_media_account.csv", sep=';', engine='python')
        self.__news = pd.read_csv("news.csv", sep=';', engine='python')
        self.__news_users = pd.read_csv("post.csv", sep=';', engine='python')
        self.__follow = pd.read_csv("userFollowingNewId.csv", sep=';', engine='python')
        self.__smoothing = laplace_smoothing
        self.__omega = omega

    def __init_params(self, test_size=0.3):
        news = self.__news[self.__news['ground_truth_label'].notnull()]
        if not len(news.index):
            return 0

        # Divide 'self.__news_users' em treino e teste
        labels = news["ground_truth_label"]
        self.__X_train_news, self.__X_test_news, _, _ = train_test_split(news, labels, test_size=test_size,
                                                                         stratify=labels)

        # Armazena em 'self.__train_news_users' as notícias compartilhadas por cada usuário
        self.__train_news_users = pd.merge(self.__X_train_news, self.__news_users, left_on="id_news",
                                           right_on="id_news")
        self.__test_news_users = pd.merge(self.__X_test_news, self.__news_users, left_on="id_news", right_on="id_news")

        # Conta a quantidade de notícias verdadeiras e falsas presentes no conjunto de treino
        self.__qtd_V = self.__news["ground_truth_label"].value_counts()[0]
        self.__qtd_F = self.__news["ground_truth_label"].value_counts()[1]

        # Filtra apenas os usuários que não estão em ambos os conjuntos de treino e teste
        self.__train_news_users = self.__train_news_users[
            self.__train_news_users["id_social_media_account"].isin(self.__test_news_users["id_social_media_account"])]

        # Inicializa os parâmetros dos usuários
        totR = 0
        totF = 0
        alphaN = totR + self.__smoothing
        umAlphaN = ((totF + self.__smoothing) / (self.__qtd_F + self.__smoothing)) * (self.__qtd_V + self.__smoothing)
        betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
        umBetaN = totF + self.__smoothing
        probAlphaN = alphaN / (alphaN + umAlphaN)
        probUmAlphaN = 1 - probAlphaN
        probBetaN = betaN / (betaN + umBetaN)
        probUmBetaN = 1 - probBetaN
        self.__users["probAlphaN"] = probAlphaN
        self.__users["probUmAlphaN"] = probUmAlphaN
        self.__users["probBetaN"] = probBetaN
        self.__users["probUmBetaN"] = probUmBetaN
        return 1

    def fit(self, test_size=0.3):

        # Etapa de treinamento: calcula os parâmetros de cada usuário a partir do Implict Crowd Signals.
        status_code = self.__init_params(test_size)
        if not status_code:
            return 0

        users_unique = self.__train_news_users["id_social_media_account"].unique()
        total = len(users_unique)

        for userId in users_unique:

            # Obtém os labels das notícias compartilhadas por cada usuário.
            newsSharedByUser = list(self.__train_news_users["ground_truth_label"].loc[
                                        self.__train_news_users["id_social_media_account"] == userId])

            # Retorna os usuários que são seguidos pelo usuário UserID.
            following_users = list(self.__follow["following_user_newId"].loc[self.__follow["user_newId"] == userId])

            # Retorna os usuários que são seguidores do usuário UserID.
            follow_userID = list(self.__follow["user_newId"].loc[self.__follow["following_user_newId"] == userId])

            for user in following_users:
                newsSharedByUserFollowing = list(self.__train_news_users["id_news"].loc[self.__train_news_users["id_social_media_account"] == user])
                for new in newsSharedByUserFollowing:
                    temp = list(self.__train_news_users["ground_truth_label"].loc[self.__train_news_users["id_news"] == new])
                    labelNewsSharedByUserFollowing = temp[0]
                    newsSharedByUser.append(labelNewsSharedByUserFollowing)

            for user in follow_userID:
                newsSharedByFollowUserID = list(self.__train_news_users["id_news"].loc[self.__train_news_users["id_social_media_account"] == user])
                for new in newsSharedByFollowUserID:
                    temp = list(self.__train_news_users["ground_truth_label"].loc[self.__train_news_users["id_news"] == new])
                    labelNewsSharedByUserFollowing = temp[0]
                    newsSharedByUser.append(labelNewsSharedByUserFollowing)

            # Calcula a matriz de opinião para cada usuário.
            totR = newsSharedByUser.count(0)
            totF = newsSharedByUser.count(1)
            alphaN = totR + self.__smoothing
            umAlphaN = ((totF + self.__smoothing) / (self.__qtd_F + self.__smoothing)) * (
                        self.__qtd_V + self.__smoothing)
            betaN = (umAlphaN * (totR + self.__smoothing)) / (totF + self.__smoothing)
            umBetaN = totF + self.__smoothing

            # Calcula as probabilidades para cada usuário.
            # Onde probAlphaN é a probabilidade de acertar dado que a notícia é verdadeira.
            # Onde probBetaN é a probabilidade de acertar dado que a notícia é fake.
            probAlphaN = alphaN / (alphaN + umAlphaN)
            probUmAlphaN = 1 - probAlphaN
            probBetaN = betaN / (betaN + umBetaN)
            probUmBetaN = 1 - probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probAlphaN"] = probAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probBetaN"] = probBetaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmAlphaN"] = probUmAlphaN
            self.__users.loc[self.__users["id_social_media_account"] == userId, "probUmBetaN"] = probUmBetaN

        accuracy = self.__assess()
        return accuracy

    def __assess(self):

        # Etapa de avaliação: avalia a notícia com base nos parâmetros de cada usuário obtidos na etapa de treinamento.
        predicted_labels = []
        unique_id_news = self.__test_news_users["id_news"].unique()

        for newsId in unique_id_news:
            # Recupera os ids de usuário que compartilharam a notícia representada por 'newsId'.
            usersWhichSharedTheNews = list(
                self.__news_users["id_social_media_account"].loc[self.__news_users["id_news"] == newsId])

            productAlphaN = 1.0
            productUmAlphaN = 1.0
            productBetaN = 1.0
            productUmBetaN = 1.0

            for userId in usersWhichSharedTheNews:
                i = self.__users.loc[self.__users["id_social_media_account"] == userId].index[0]
                productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
                productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

            # Inferência bayesiana
            reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
            reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

            if reputation_news_tn >= reputation_news_fn:
                predicted_labels.append(0)
            else:
                predicted_labels.append(1)

        # Mostra os resultados da matriz de confusão e acurácia.
        gt = self.__X_test_news["ground_truth_label"].tolist()
        accuracy = accuracy_score(gt, predicted_labels)
        return accuracy

    def predict(self, id_news):

        # Classifica uma notícia usando o ICS
        usersWhichSharedTheNews = list(
            self.__news_users["id_social_media_account"].loc[self.__news_users["id_news"] == id_news])
        productAlphaN = 1.0
        productUmAlphaN = 1.0
        productBetaN = 1.0
        productUmBetaN = 1.0

        for userId in usersWhichSharedTheNews:
            i = self.__users.loc[self.__users["id_social_media_account"] == userId].index[0]
            productAlphaN = productAlphaN * self.__users.at[i, "probAlphaN"]
            productUmBetaN = productUmBetaN * self.__users.at[i, "probUmBetaN"]

        # Inferência bayesiana
        reputation_news_tn = (self.__omega * productAlphaN * productUmAlphaN) * 100
        reputation_news_fn = ((1 - self.__omega) * productBetaN * productUmBetaN) * 100

        # Calculando o grau de probabilidade da predição.
        total = reputation_news_tn + reputation_news_fn

        if reputation_news_tn >= reputation_news_fn:
            prob = reputation_news_tn / total
            # Notícia classificada como legítima.
            return 0, prob
        else:
            prob = reputation_news_fn / total
            # Notícia classificada como fake.
            return 1, prob


ics = ICS()
ics.fit()
media = 0.0
tamanho_de_teste = 0.3
quantidade = 20
for i in range(quantidade):
    acuracia = ics.fit(tamanho_de_teste)
    print("Acurácia", i+1, "=", acuracia)
    media += acuracia / quantidade
print("media = ", media)


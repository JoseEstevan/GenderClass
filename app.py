import streamlit as st 
from sklearn.externals import joblib
import time
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import hyperlink


gender_vectorizer = open("gender_vectorizer.pkl",'rb')
gender_cv = joblib.load(gender_vectorizer)

gender_nv_model = open("naivebayesgendermodel.pkl",'rb')
gender_clf = joblib.load(gender_nv_model)


def predict_gender(data):
	vect = gender_cv.transform(data).toarray()
	result = gender_clf.predict(vect)
	return result


def load_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

def load_icon(icon_name):
    st.markdown('<i class="material-icons">{}</i>'.format(icon_name), unsafe_allow_html=True)


def load_images(file_name):
	img = Image.open(file_name)
	return st.image(img,width=300)


def main():
	st.title("Classificação de Gênero com ML")
	html_temp = """
	<div style="background-color:tomato;padding:10px">
	<h2 style="color:white;text-align:center;">GenderClass App</h2>
	</div>
	"""
	st.markdown(html_temp,unsafe_allow_html=True)
	load_css('icone.css')
	load_icon('people')

	name = st.text_input("Digite o nome" )
	if st.button("Classificar"):
		result = predict_gender([name])
		if result[0] == 0:
			prediction = 'Mulher'
			c_img = 'mulher.png'
		elif result[0] == 1:
			prediction = 'Homem'
			c_img = 'homem.png'
        
		st.success('{} foi classificado como {}'.format(name.title(),prediction))
		load_images(c_img)
        


if __name__ == '__main__':
	main()

if st.checkbox("Sobre"):
    
    st.text("GenderClass é um app de classificação de gênero com base no nome\nFaz uso da biblioteca scikit-learn e da linguagem Python")
    st.subheader('Redes Sociais')
        
    medium = hyperlink.parse(u'https://medium.com/@joseestevan')
    st.markdown(medium.to_text())
    
    link = hyperlink.parse(u'https://www.linkedin.com/in/joseestevan/')
    st.markdown(link.to_text())  

    git = hyperlink.parse(u'https://github.com/JoseEstevan')
    st.markdown(git.to_text())


st.markdown('')
st.markdown('Obs: Se a classificação estiver errada tente com o sobrenome, o modelo está sendo revisado por conta de alguns erros')

st.subheader('By: José Estevan')

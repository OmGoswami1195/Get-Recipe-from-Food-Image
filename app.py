import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url("https://images.unsplash.com/photo-1539056276907-dc946d5098c9?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: cover;
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Load the trained model
model = tf.keras.models.load_model('foody.h5')

# Define the target image size
target_size = (224, 224)

class_labels = {
    0: 'burger',
    1: 'dal_makhani',
    2: 'jalebi',
    3: 'pakode',
    4: 'pav_bhaji',
    5: 'kadai_paneer',
    6: 'momos',
    7: 'samosa',
    8: 'masala_dosa',
    9: 'kaathi_rolls',
    10: 'butter_naan',
    11: 'pizza',
    12: 'chai',
    13: 'kulfi',
    14: 'dhokla',
    15: 'paani_puri',
    16: 'idli',
    17: 'fried_rice',
    18: 'chole_bhature',
    19: 'chapati'
}

recipes = {
    'burger': '''
        Burger Recipe:
        Ingredients:
        - 1 pound chicken
        - Salt and pepper to taste
        - 4 hamburger buns
        - 4 slices of cheese
        - Lettuce, tomato, onion for toppings
        - Ketchup and mustard

        Instructions:
        1. Divide chicken into four equal parts and shape them into patties.
        2. Season each patty with salt and pepper.
        3. Grill the patties to your preferred doneness.
        4. Place a slice of cheese on each patty during the last minute of grilling.
        5. Toast the hamburger buns on the grill.
        6. Assemble the burgers by placing the patties on the buns and adding toppings.
        7. Add ketchup and mustard to taste.
        8. Serve the burgers hot and enjoy!
    ''',

    'dal_makhani': '''
        Dal Makhani Recipe:
        Ingredients:
        - 1 cup black gram (urad dal)
        - 1/4 cup kidney beans (rajma)
        - 1 large onion, finely chopped
        - 2 tomatoes, pureed
        - 1 tablespoon ginger-garlic paste
        - 2 tablespoons butter
        - 1/4 cup cream
        - 1 teaspoon garam masala
        - 1 teaspoon cumin powder
        - 1 teaspoon coriander powder
        - 1 teaspoon red chili powder
        - Salt to taste

        Instructions:
        1. Soak black gram and kidney beans overnight. Cook until soft.
        2. In a pan, heat butter and sauté chopped onions until golden brown.
        3. Add ginger-garlic paste, tomato puree, and spices. Cook until oil separates.
        4. Add cooked dal and kidney beans. Simmer for an hour.
        5. Add cream and garam masala. Adjust salt and spices.
        6. Serve hot with rice or naan.
    ''',

    'jalebi': '''
        Jalebi Recipe:
        Ingredients:
        - 1 cup all-purpose flour (maida)
        - 1 cup curd (yogurt)
        - A pinch of saffron strands
        - 1 cup sugar
        - 1/2 cup water
        - Ghee for frying

        Instructions:
        1. Make a smooth batter using all-purpose flour and curd. Let it ferment for a few hours.
        2. Prepare sugar syrup by dissolving sugar in water. Add saffron strands.
        3. Heat ghee in a pan for frying.
        4. Pour the batter into a jalebi dispenser and make circular shapes in hot ghee.
        5. Fry until golden brown, then dip in sugar syrup.
        6. Allow jalebis to soak in syrup and serve warm.
    ''',
    'pakode': '''
        Pakode Recipe:
        Ingredients:
        - 1 cup gram flour (besan)
        - 1 medium-sized potato, thinly sliced
        - 1 onion, thinly sliced
        - 1/2 cup spinach, chopped
        - 1/2 cup cauliflower florets
        - 1/2 cup water
        - 1 teaspoon red chili powder
        - 1/2 teaspoon turmeric powder
        - Salt to taste
        - Oil for frying

        Instructions:
        1. In a bowl, mix gram flour with water to make a smooth batter.
        2. Add sliced potatoes, onions, spinach, cauliflower, red chili powder, turmeric powder, and salt to the batter.
        3. Heat oil in a pan for frying.
        4. Drop spoonfuls of the batter into the hot oil and fry until golden brown.
        5. Remove pakoras from the oil and place them on a paper towel to absorb excess oil.
        6. Serve hot with chutney or ketchup.
    ''',

    'pav_bhaji': '''
        Pav Bhaji Recipe:
        Ingredients:
        - 4 medium-sized potatoes, boiled and mashed
        - 1 cup mixed vegetables (peas, carrots, capsicum), finely chopped
        - 1 large onion, finely chopped
        - 2 tomatoes, finely chopped
        - 1/4 cup pav bhaji masala
        - 1 teaspoon red chili powder
        - 1/2 teaspoon turmeric powder
        - Salt to taste
        - Butter for cooking
        - Pav (bread rolls) for serving

        Instructions:
        1. In a pan, sauté onions until golden brown.
        2. Add chopped tomatoes and cook until they turn soft.
        3. Add mixed vegetables and cook until they are tender.
        4. Add mashed potatoes, pav bhaji masala, red chili powder, turmeric powder, and salt.
        5. Mash the mixture and cook until the bhaji is well blended.
        6. Toast pav with butter on a griddle.
        7. Serve hot bhaji with buttered pav.
    ''',

    'kadai_paneer': '''
        Kadai Paneer Recipe:
        Ingredients:
        - 1 cup paneer cubes
        - 1 cup bell peppers, sliced
        - 1 cup tomatoes, chopped
        - 1 large onion, finely chopped
        - 2 tablespoons kadai masala
        - 1 teaspoon ginger-garlic paste
        - 1/2 teaspoon turmeric powder
        - 1/2 teaspoon red chili powder
        - 1/2 teaspoon garam masala
        - Salt to taste
        - 2 tablespoons oil
        - Fresh coriander for garnish

        Instructions:
        1. Heat oil in a kadai or wok. Add ginger-garlic paste and sauté until fragrant.
        2. Add chopped onions and cook until they become translucent.
        3. Add sliced bell peppers and cook until they are slightly tender.
        4. Add chopped tomatoes and cook until they are soft.
        5. Add kadai masala, turmeric powder, red chili powder, garam masala, and salt. Mix well.
        6. Add paneer cubes and cook until they are heated through.
        7. Garnish with fresh coriander and serve hot.
    ''',

    'momos': '''
        Momos Recipe:
        Ingredients:
        - 1 cup all-purpose flour
        - 1 cup minced chicken or vegetables
        - 1/2 cup cabbage, finely chopped
        - 1/4 cup spring onions, chopped
        - 1 teaspoon ginger-garlic paste
        - 1 teaspoon soy sauce
        - 1/2 teaspoon black pepper powder
        - Salt to taste
        - Oil for greasing and steaming

        For dipping sauce:
        - 2 tablespoons soy sauce
        - 1 tablespoon vinegar
        - 1 teaspoon red chili sauce

        Instructions:
        1. Mix all-purpose flour with water to make a smooth dough. Rest for 30 minutes.
        2. Roll the dough into small circles and fill them with a mixture of minced chicken or vegetables, cabbage, spring onions, ginger-garlic paste, soy sauce, black pepper, and salt.
        3. Shape them into momos and place them in a steamer.
        4. Steam for about 15-20 minutes until cooked.
        5. For the dipping sauce, mix soy sauce, vinegar, and red chili sauce.
        6. Serve momos hot with the dipping sauce.
    ''',

    'samosa': '''
        Samosa Recipe:
        Ingredients:
        - 2 cups all-purpose flour
        - 1/4 cup ghee
        - 1 cup boiled and mashed potatoes
        - 1/2 cup peas
        - 1 teaspoon cumin seeds
        - 1 teaspoon coriander powder
        - 1/2 teaspoon turmeric powder
        - 1/2 teaspoon red chili powder
        - 1/2 teaspoon garam masala
        - Salt to taste
        - Oil for frying

        Instructions:
        1. Mix all-purpose flour with ghee and knead it into a stiff dough. Rest for 30 minutes.
        2. In a pan, heat oil and add cumin seeds. Sauté until they splutter.
        3. Add boiled and mashed potatoes, peas, coriander powder, turmeric powder, red chili powder, garam masala, and salt. Mix well.
        4. Roll the dough into small circles and fill them with the potato mixture.
        5. Shape them into samosas and deep fry until golden brown.
        6. Serve hot with mint chutney.
    ''',
    'masala_dosa': '''
        Masala Dosa Recipe:
        Ingredients:
        For Dosa:
        - 2 cups dosa rice
        - 1/2 cup urad dal (black gram)
        - 1/4 teaspoon fenugreek seeds
        - Salt to taste

        For Potato Filling:
        - 4 large potatoes, boiled and mashed
        - 1 large onion, finely chopped
        - 2 green chilies, chopped
        - 1/2 teaspoon mustard seeds
        - 1/2 teaspoon cumin seeds
        - A pinch of asafoetida (hing)
        - Curry leaves
        - Salt to taste

        Instructions:
        1. Soak dosa rice, urad dal, and fenugreek seeds separately for 6 hours.
        2. Grind them into a smooth batter, mix salt, and ferment overnight.
        3. For the potato filling, heat oil, add mustard seeds, cumin seeds, asafoetida, curry leaves, and chopped onions. Sauté until golden brown.
        4. Add mashed potatoes, chopped green chilies, and salt. Mix well and set aside.
        5. Heat a dosa griddle, pour batter, and spread it thin.
        6. Place a spoonful of potato filling in the center and fold the dosa.
        7. Serve hot with coconut chutney and sambar.
    ''',

    'kaathi_rolls': '''
        Kaathi Rolls Recipe:
        Ingredients:
        For Paratha:
        - 2 cups whole wheat flour
        - Water for kneading
        - Salt to taste

        For Filling:
        - 1 cup boneless chicken, marinated
        - 1 cup bell peppers, thinly sliced
        - 1 cup onions, thinly sliced
        - 1 teaspoon ginger-garlic paste
        - 1 teaspoon garam masala
        - 1/2 teaspoon red chili powder
        - Salt to taste
        - Oil for cooking

        Instructions:
        1. Knead whole wheat flour with water and salt to make a soft dough. Set aside.
        2. Cook marinated chicken until tender. Shred it into small pieces.
        3. In a pan, heat oil, add ginger-garlic paste, sliced onions, and bell peppers. Sauté until soft.
        4. Add shredded chicken, garam masala, red chili powder, and salt. Cook until well combined.
        5. Roll the dough into parathas and cook on a griddle.
        6. Place the chicken filling on the paratha and roll it up.
        7. Serve hot with mint chutney.
    ''',

    'butter_naan': '''
        Butter Naan Recipe:
        Ingredients:
        - 2 cups all-purpose flour
        - 1/2 cup yogurt
        - 1 teaspoon baking powder
        - 1/2 teaspoon baking soda
        - 1/2 teaspoon sugar
        - Salt to taste
        - 2 tablespoons ghee
        - Water for kneading
        - Butter for brushing

        Instructions:
        1. Mix all-purpose flour, yogurt, baking powder, baking soda, sugar, salt, and ghee.
        2. Knead into a soft dough using water. Let it rest for 2 hours.
        3. Divide the dough into small balls and roll them into naans.
        4. Cook the naans on a hot griddle or tandoor until they puff up and have brown spots.
        5. Brush with butter and serve hot with curry or raita.
    ''',

    'pizza': '''
        Pizza Recipe:
        Ingredients:
        For Pizza Dough:
        - 2 1/4 teaspoons active dry yeast
        - 1 teaspoon sugar
        - 1 cup warm water
        - 2 1/2 cups all-purpose flour
        - 1 teaspoon salt
        - 2 tablespoons olive oil

        For Toppings:
        - Pizza sauce
        - Mozzarella cheese
        - Bell peppers, sliced
        - Mushrooms, sliced
        - Olives, sliced
        - Pepperoni slices (optional)

        Instructions:
        1. Dissolve yeast and sugar in warm water. Let it sit for 5 minutes.
        2. Mix flour and salt in a bowl. Add the yeast mixture and olive oil. Knead into a smooth dough.
        3. Let the dough rise for 1-2 hours.
        4. Roll out the dough into a pizza shape.
        5. Spread pizza sauce and add toppings of your choice.
        6. Bake in a preheated oven at 475°F (245°C) for 12-15 minutes or until the crust is golden and cheese is bubbly.
        7. Slice and enjoy your homemade pizza!
    ''',

    'chai': '''
        Chai Recipe:
        Ingredients:
        - 2 cups water
        - 1 cup milk
        - 2 teaspoons loose tea leaves
        - 2-3 teaspoons sugar (adjust to taste)
        - 1-2 slices of fresh ginger
        - 2-3 cardamom pods
        - 1-2 cinnamon sticks

        Instructions:
        1. In a saucepan, bring water, milk, ginger, cardamom, and cinnamon to a boil.
        2. Add tea leaves and sugar. Simmer for 2-3 minutes.
        3. Strain the chai into cups.
        4. Serve hot and enjoy the aromatic Indian chai!
    ''',
    'kulfi': '''
        Kulfi Recipe:
        Ingredients:
        - 2 cups full-fat milk
        - 1/2 cup condensed milk
        - 1/4 cup powdered sugar
        - 1/2 cup mixed nuts (almonds, pistachios), chopped
        - A pinch of saffron strands (optional)
        - Cardamom powder

        Instructions:
        1. In a pan, boil the milk until it reduces to half.
        2. Add condensed milk, powdered sugar, saffron strands, and cardamom powder. Mix well.
        3. Remove from heat and let it cool.
        4. Add chopped nuts to the mixture.
        5. Pour the mixture into kulfi molds and freeze for 6-8 hours or overnight.
        6. Unmold the kulfi and serve chilled.
    ''',

    'dhokla': '''
        Dhokla Recipe:
        Ingredients:
        - 1 cup gram flour (besan)
        - 1/2 cup semolina (sooji)
        - 1 cup yogurt
        - 1/2 teaspoon ginger-green chili paste
        - 1/4 teaspoon turmeric powder
        - 1 teaspoon eno (fruit salt)
        - Salt to taste

        For Tempering:
        - 2 tablespoons oil
        - 1 teaspoon mustard seeds
        - 1 teaspoon sesame seeds
        - Curry leaves
        - 2 green chilies, slit

        Instructions:
        1. Mix gram flour, semolina, yogurt, ginger-green chili paste, turmeric powder, and salt to make a smooth batter.
        2. Add eno and mix well. Pour the batter into a greased plate.
        3. Steam the batter for 15-20 minutes until cooked.
        4. For tempering, heat oil, add mustard seeds, sesame seeds, curry leaves, and green chilies. Pour over the dhokla.
        5. Cut into squares and serve with mint chutney.
    ''',

    'paani_puri': '''
        Pani Puri Recipe:
        Ingredients:
        For Puri:
        - 1 cup semolina (sooji)
        - 1/4 cup all-purpose flour
        - Water for kneading
        - Oil for frying

        For Pani:
        - 1 cup mint leaves
        - 1/2 cup coriander leaves
        - 2 green chilies
        - 1 teaspoon tamarind pulp
        - 1 teaspoon chaat masala
        - 1/2 teaspoon black salt
        - 1/2 teaspoon cumin powder
        - Salt to taste
        - Water

        For Filling:
        - Boiled and mashed potatoes
        - Boiled chickpeas
        - Tamarind chutney

        Instructions:
        1. Mix semolina, all-purpose flour, and water to make a stiff dough. Rest for 30 minutes.
        2. Roll the dough into small balls and fry until they puff up.
        3. Blend mint leaves, coriander leaves, green chilies, tamarind pulp, chaat masala, black salt, cumin powder, and salt to make pani.
        4. Make a hole in each puri, fill with mashed potatoes, chickpeas, and tamarind chutney.
        5. Pour the prepared pani into the puris and serve immediately.
    ''',

    'idli': '''
        Idli Recipe:
        Ingredients:
        - 2 cups idli rice
        - 1/2 cup urad dal (black gram)
        - 1/4 teaspoon fenugreek seeds
        - Salt to taste

        Instructions:
        1. Soak idli rice, urad dal, and fenugreek seeds separately for 6 hours.
        2. Grind them into a smooth batter, mix salt, and ferment overnight.
        3. Grease idli molds and pour the batter into each mold.
        4. Steam the idlis for 10-12 minutes until cooked.
        5. Serve hot with coconut chutney and sambar.
    ''',
    'fried_rice': '''
        Fried Rice Recipe:
        Ingredients:
        - 2 cups cooked basmati rice (preferably cold)
        - 1 cup mixed vegetables (carrots, peas, corn, bell peppers)
        - 1/2 cup diced tofu or cooked chicken (optional)
        - 2 eggs, beaten
        - 3 tablespoons soy sauce
        - 1 tablespoon oyster sauce
        - 1 tablespoon sesame oil
        - 2 tablespoons vegetable oil
        - 2 cloves garlic, minced
        - 1 teaspoon ginger, grated
        - Spring onions for garnish
        - Salt and pepper to taste

        Instructions:
        1. Heat vegetable oil in a pan. Add minced garlic and grated ginger, sauté until fragrant.
        2. Add mixed vegetables and tofu or chicken if using. Cook until vegetables are tender.
        3. Push the vegetables to one side, pour beaten eggs, and scramble.
        4. Add cold cooked rice, soy sauce, oyster sauce, sesame oil, salt, and pepper. Mix well.
        5. Cook for a few minutes until everything is heated through.
        6. Garnish with chopped spring onions and serve hot.

    ''',

    'chole_bhature': '''
        Chole Bhature Recipe:
        Ingredients:
        For Chole:
        - 1 cup chickpeas, soaked overnight
        - 1 large onion, finely chopped
        - 2 large tomatoes, pureed
        - 1/2 cup yogurt
        - 1 tablespoon ginger-garlic paste
        - 1 tablespoon chole masala
        - 1 teaspoon cumin powder
        - 1 teaspoon coriander powder
        - 1/2 teaspoon turmeric powder
        - 1/2 teaspoon red chili powder
        - 1/2 teaspoon garam masala
        - Fresh coriander leaves for garnish
        - Salt to taste
        - Oil for cooking

        For Bhature:
        - 2 cups all-purpose flour
        - 1/2 cup semolina (sooji)
        - 1/2 cup yogurt
        - 1 teaspoon sugar
        - 1/2 teaspoon baking powder
        - 1/4 teaspoon baking soda
        - Salt to taste
        - Water for kneading
        - Oil for frying

        Instructions:
        1. Cook soaked chickpeas until tender. In a pan, heat oil, add chopped onions, and sauté until golden brown.
        2. Add ginger-garlic paste, pureed tomatoes, and spices. Cook until the oil separates.
        3. Add cooked chickpeas, yogurt, and water. Simmer until the chole thickens.
        4. For bhature, mix all-purpose flour, semolina, yogurt, sugar, baking powder, baking soda, and salt. Knead into a soft dough.
        5. Divide the dough into small balls and roll them into bhaturas. Fry until puffed and golden.
        6. Serve hot chole with bhature, garnished with fresh coriander leaves.

    ''',

    'chapati': '''
        Chapati Recipe:
        Ingredients:
        - 2 cups whole wheat flour
        - Water for kneading
        - Salt to taste

        Instructions:
        1. In a bowl, mix whole wheat flour and salt.
        2. Gradually add water and knead into a smooth dough.
        3. Divide the dough into small balls.
        4. Roll each ball into a flat, round chapati.
        5. Heat a griddle or tawa, place the chapati, and cook until bubbles appear.
        6. Flip and cook the other side until golden brown spots appear.
        7. Serve hot with your favorite curry or accompaniment.

    '''
}


# Function to preprocess and predict the image
def predict_image(img_path):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    prediction = model.predict(img_array)
    return prediction

# Streamlit app
st.title("Get Your Recipe From Food Images...")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Make a prediction
    prediction = predict_image(uploaded_file)

    # Get the predicted class label
    class_index = np.argmax(prediction)
    confidence = prediction[0, class_index]
    
    # Map the class index to the corresponding label
    class_label = class_labels.get(class_index, f'Unknown Class ({class_index})')

    # Display the result
    st.write(f"Prediction: {class_label}, Confidence: {confidence:.2%}")

    # Display recipe if available
    recipe = recipes.get(class_label)
    if recipe:
        st.write(f"Recipe: {recipe}")
    else:
        st.write("Recipe not available.")

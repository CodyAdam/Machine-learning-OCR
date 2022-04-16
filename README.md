# Machine-learning-OCR
Machine learning Optical Character Recognition (OCR) project. Read the time on a digital clock but inputting an image.


 <div class="img-container">
   <style>
    html,
    body {
      width: 100%;
      height: 100%;
      font-family: Century Gothic, CenturyGothic, AppleGothic, sans-serif;
    }
    img {
      border-radius: 17px;
      width: auto;
      height: 350px;
      margin-left: 30px;
      margin-right: 30px;
    }
    .img-container {
      width: 100%;
      height: 350px;
      margin-bottom: 50px;
      display: inline-block;
    }
    .left {
      float: left;
    }
    .right {
      float: right;
    }
    p {
      font-size: 1.2rem;
      margin: 0;
      box-sizing: border-box;
    }
  </style>
      <img src="Original.png" alt="Original.png is missing" class="left" />
      <h1 style="text-align: left">Image à analyser</h1>
      <p style="text-align: left"></p>
    </div>
    <div class="img-container">
      <img
        src="HomographyMatches.png"
        alt="HomographyMatches.png is missing"
        class="right"
      />
      <h1 style="text-align: right">Teste homographique</h1>
      <p style="text-align: right">
        Recherche des similarités avec<br />
        une image de référence
      </p>
    </div>
    <div class="img-container">
      <img
        src="WrapedImage.png"
        alt="WrapedImage.png is missing"
        class="left"
      />
      <h1 style="text-align: left">Résultat de l'homographie</h1>
      <p style="text-align: left">
        Si l'homographie est un succès alors l'image <br />
        est rognée et transformée selon la référence homographie <br />
        Sinon l'image reste l'image d'entrée
      </p>
    </div>
    <div class="img-container">
      <img src="BinaryMask.png" alt="BinaryMask.png is missing" class="right" />
      <h1 style="text-align: right">Création du masque</h1>
      <p style="text-align: right">
        On crée une image binaire à partir de l'image<br />
        précédente en fonction d'un seuil de couleur
      </p>
    </div>
    <div class="img-container">
      <img
        src="FinalResult.png"
        alt="FinalResult.png is missing"
        class="left"
      />
      <h1 style="text-align: left">Analyse machine learning</h1>
      <p style="text-align: left">
        On détoure chaque élément de l'image binaire, <br />
        puis on compare ces formes avec <br />
        une banque d'images de chiffre<br />
        Ensuite l'algorithme renvoie le digit correspondant <br />
      </p>
    </div>
```html
<html>
	<head>
		<title>Generative View Synthesis: From Single-view Semantics to Novel-view Images</title>
		<meta property="og:title" content="Generative View Synthesis: from Single View Semantics to Novel-view Images" />
		<meta property="og:description" content="Habtegebrial et al. 2020" />
  </head>

  <body>
    <br>
          <center>
          	<span style="font-size:36px">Generative View Synthesis: <br> From Single-view Semantics to Novel-view Images</span>
	  		  <table align=center width=800px>
	  			  <tr>
	  	              <td align=center width=300px>
	  					<center>
							<span style="font-size:20px"><a href="http://tedyhabtegebrial.github.io/">Tewodros A. Habtegebrial<sup>1,4</sup> </a></span>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=150px>
	  					<center>
							<span style="font-size:20px"><a href="http://varunjampani.github.io/">Varun Jampani<sup>2</sup></a></span>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=150px>
	  					<center>
							<span style="font-size:20px"><a href="http://alumni.soe.ucsc.edu/~orazio/">Orazio Gallo<sup>3</sup></a></span>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=150px>
	  					<center>
	  						<span style="font-size:20px"><a href="https://av.dfki.de/members/stricker/">Didier Stricker<sup>1,4</sup></a></span>
		  		  		</center>
		  		  	  </td>
		  		  </tr>
					<td align=center width=200px> <sup>1</sup> TU Kaiserslautern</td>
					<td align=center width=200px> <sup>2</sup> Google Research</td>
					<td align=center width=200px> <sup>3</sup> NVIDIA</td>
					<td align=center width=200px> <sup>4</sup> DFKI Kaiserslautern</td>

	  			  <tr>
		  		  </tr>
			  </table>
			<!--
	  		  <table align=center width=600px>
	  			  <tr>
	  	              <td align=center width=120px>
	  					<center>
	  						<span style="font-size:24px"><a href='https://arxiv.org/abs/1904.11486'>[Paper]</a></span>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=120px>
	  					<center>
	  						<span style="font-size:24px"><a href='https://github.com/adobe/antialiased-cnns'>[GitHub]</a></span><br>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=120px>
	  					<center>
	  						<span style="font-size:24px"><a href='https://www.youtube.com/watch?v=HjewNBZz00w'>[Talk]</a></span><br>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=120px>
	  					<center>
	  						<span style="font-size:24px"><a href='https://www.dropbox.com/s/bzo8kia5si811tm/antialiasing_bair.pptx?dl=0'>[Slides]</a></span><br>
		  		  		</center>
		  		  	  </td>
	  	              <td align=center width=120px>
	  					<center>
	  						<span style="font-size:24px"><a href="https://www.dropbox.com/s/dhf2gqt14sq76q7/poster_icml.pdf?dl=0">[Poster]</a></span><br>
		  		  		</center>
		  		  	  </td>
		  		  	 </tr>
	  			  <tr>
			  </table>
			-->
          </center>
		  <br>
          <center>
  	<table align=center width=750px>
			<tr>
				<td align=center width=150px>Input Semantics</td>
  	            <td width=35px>&emsp;</td>
				<td align=center width=150px><p style="border:2px; border-style:solid; border-color:#FF0000;"> GVSNet (Ours)</p></td>
				<td width=35px>&emsp;</td>
				<td align=center width=150px>SPADE[1]+SM[2]</td>
				<td width=35px>&emsp;</td>
				<td align=center width=150px>SPADE[1]+CVS[3]</td>
      </tr>
			<td width=35px> &emsp; </td>
			<tr>
				<td width=150px>
					<center>
						<img class="round" style="width:200px" src="./resources/video/carla/circle_r_0_25/0_input_sem.png"/>
		  		</center>
				</td>
				<td width=35px> &emsp; </td>
    	  <td width=150px>
					<center>
						<img class="round" style="border:2px solid red;" width="200" src="./resources/video/carla/circle_r_0_25/0_Ours_gif.gif"/>
	  			</center>
  	    </td>
				<td width=35px> &emsp; </td>
        <td width=150px>
  			<center>
  	    	<img class="round" style="width:200px" src="./resources/video/carla/circle_r_0_25/0_SPADE+SM_gif.gif"/>
				</center>
				</td>
  	    	<td width=35px>&emsp;</td>
        <td width=150px>
  				<center>
						<img class="round" style="width:200px" src="./resources/video/carla/circle_r_0_25/0_SPADE+CVS_gif.gif"/>
	  			</center>
  	    </td>
			</tr>
			<td width=35px> &emsp; </td>
			<tr>
				<td width=150px>
					<center>
						<img class="round" style="width:200px" src="./resources/video/carla/lateral_x__0_4_to_0_4/3108_input_sem.png"/>
		  		</center>
				</td>
				<td width=35px> &emsp; </td>
    	  <td width=150px>
					<center>
						<img class="round" style="border:2px solid red;" width="200" src="./resources/video/carla/lateral_x__0_4_to_0_4/3108_Ours_gif.gif"/>
	  			</center>
  	    </td>
				<td width=35px> &emsp; </td>
        <td width=150px>
  			<center>
  	    	<img class="round" style="width:200px" src="./resources/video/carla/lateral_x__0_4_to_0_4/3108_SPADE+SM_gif.gif"/>
				</center>
				</td>
  	    	<td width=35px>&emsp;</td>
        <td width=150px>
  				<center>
						<img class="round" style="width:200px" src="./resources/video/carla/lateral_x__0_4_to_0_4/3108_SPADE+CVS_gif.gif"/>
	  			</center>
  	    </td>
			</tr>
		</table>

  	<table align=center width=850px>
			<tr>
				<td>
					<!-- <center> -->
					<!-- Describe GVS and out solution.-->
					<!-- </center> -->
				</td>
			</tr>
		</table>
	</center>

          <hr>

<!--   		  <br>
  		  <table align=center width=800px>
  			  <tr>
  	              <td width=400px>
  					<center>
  	                	<a href="./resources/fig1d.jpeg"><img class="" src = "./resources/fig1d.jpeg" height="400px"></img></href></a><br>
					</center>
  	              </td>
                </tr>
  		  </table>
		  <hr> -->
				<table align="center" width=850px>
					<center> Accepted at NeurIPS-2020 </center>
				</table>

  		  <table align=center width=850px>
	  		  <center><h1>Abstract</h1></center>
	  		  <tr>
	  		  	<td>
							<div align="justify">
							Content creation, central to applications such as virtual reality, can be a tedious and time-consuming.
							Recent image synthesis methods simplify this task by offering tools to generate new views from as little
							as a single input image, or by converting a semantic map into a photorealistic image. We propose to push
							the envelope further, and introduce Generative View Synthesis (GVS), which can synthesize multiple photorealistic views
							of a scene given a single semantic map. We show that the sequential application of existing techniques, e.g., semantics-to-image
							translation followed by monocular view synthesis, fail at capturing the scene's structure. In contrast, we solve the semantics-to-image
							translation in concert with the estimation of the 3D layout of the scene, thus producing geometrically consistent novel views that preserve
							semantic structures. We first lift the input 2D semantic map onto a 3D layered representation of the scene in feature space, thereby preserving
							the semantic labels of 3D geometric structures. We then project the layered features onto the target views to generate the final novel-view images.
							We verify the strengths of our method and compare it with several advanced baselines on three different datasets. Our approach also allows for style
							manipulation and image editing operations, such as the addition or removal of objects, with simple manipulations of the input style images and semantic maps respectively.

							</div>
	  		    </td>
	  		  </tr>
			</table>
  		  <br>
		  <hr>

  		  <table align=center width=700px>
	  		  <center><h1>Method overview</h1></center>
  			  <tr>
  	              <td align=center width=700px>
  					<center>
						  <td><img class="round" style="width:800px" src="./resources/GVSNet.png"/></td>
	  		  		</center>
			  </tr>
		  </center>
		  </table>
  		  <table align=center width=850px>
		  	<center>
		  		<tr>
		  			<td>
				  	<!-- Method overview -->
		  			</td>
		  		</tr>
		  </center>
		  </table>
		  <br>
        <hr>
        <center><h1>Supplementary Video</h1></center>
        <p align="center">
				<iframe width="660" height="395" src="https://www.youtube.com/embed/qz2yX8TIzDk" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen align="center"></iframe></p>

<!--   		  <table align=center width=700px>
  			  <tr>
  	              <td align=center width=700px>
  					<center>
						  <td><img class="round" style="width:800px" src="./resources/fig1e.jpg"/></td>
	  		  		</center>
			  </tr>
		  </table>
		  <br> -->

		<!--
		  <hr>

        <center><h1>Supplementary Video</h1></center>
        <p align="center">
		<iframe width="660" height="395" src="https://youtu.be/kyi5W5rKOnw" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen align="center"></iframe></p>


  		  <table align=center width=800px>
			  <br>
			  <tr><center>
				<span style="font-size:28px">&nbsp;<a href='https://www.dropbox.com/s/bzo8kia5si811tm/antialiasing_bair.pptx?dl=0'>[Slides]</a>
		  </table>
        <hr>
		-->
	  <!-- NETWORK ARCHITECTURE, TRY THE MODEL -->
	  <!--
 		<center><h1>Code and Antialiased Models</h1></center>

  		  <table align=center width=420px>
		  	<center>
		  		<tr>
		  			<td>
		  				<b>ImageNet Classification (shift-invariance vs accuracy)</b>
		  			</td>
		  		</tr>
		  </center>
		  </table>
  		  <table align=center width=400px>
  			  <tr>
  	              <td align=center width=400px>
  					<center>
						  <td><img class="round" style="width:450px" src="./resources/imagenet_ind2_noalex.jpg"/></td>

	  		  		</center>
			  </tr>
		  </table>

  		  <table align=center width=850px>
		  	<center>
		  		<tr>
		  			<td>
		  	As designed, adding low-pass filtering increases <b>shift-invariance (y-axis)</b>. Surprisingly, we also observe increases in <b>accuracy (x-axis)</b>, across architectures, as well as increased robustness. We have pretrained anti-aliased models, along with instructions for making your favorite architecture more shift-invariant.
		  			</td>
		  		</tr>
		  </center>
		  </table>
		   -->
  		  <table align=center width=800px>
			  <br>
			  <tr><center>
				<span style="font-size:28px">&nbsp;<a href='https://github.com/tedyhabtegebrial/gvsnet'>[GitHub]</a> coming soon
		  </table>

<!-- <a href="http://www.eecs.berkeley.edu/~rich.zhang/projects/2016_colorization/files/demo_v0/colorization_release_v0.caffemodel">[Model 129MB]</span> -->

      	  <br>
		  <hr>

  		  <!-- <table align=center width=550px> -->
  		  <table align=center width=490px>
	 		<center><h1>Paper and Supplementary Material</h1></center>
  			  <tr>
				  <!-- <td><a href="./resources/camera-ready.pdf"><img class="layered-paper-big" style="height:175px" src="./resources/paper.png"/></a></td> -->
				  <td><a href="https://arxiv.org/abs/2008.09106"><img class="layered-paper-big" style="height:175px" src="./resources/paper.png"/></a></td>
				  <td><span style="font-size:14pt">Tewodros Habtegebrial, Varun Jampani, Orazio Gallo, Didier Stricker.<br>
				  <b>Generative View Synthesis: <em>From Single-view Semantics to Novel-view Images  </em></b><br>
				  Arxiv Preprint, 2020. <a href="https://arxiv.org/abs/2008.09106">Link</a><br>
				  <span style="font-size:4pt"><a href=""><br></a>
				  </span>
				  </td>
  	              </td>
              </tr>
  		  </table>
		  <br>

		  <table align=center width=600px>
			  <tr>
				  <td><span style="font-size:14pt"><center>
				  	<a href="./resources/arxiv_bib.txt">[Bibtex]</a>
  	              </center></td>
              </tr>
  		  </table>

		  <hr>
		  <br>

<table align=center width=900px>
	<tr> <td> <span style="font-size:14pt"> References<center> </td> </tr>
	<tr> <td> [1] SPADE: <em>Semantic Image Synthesis with Spatially-Adaptive Normalization</em>, Park et al. <a href="https://arxiv.org/abs/1903.07291">link</a></td> </tr>
	<tr> <td> [2] SM: <em> Stereo Magnification: Learning View Synthesis using Multiplane Images</em>, Zhou et al. <a href="https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/"> link </a> </td> </tr>
	<tr> <td> [3] CVS: <em> Monocular Neural Image Based Rendering with Continuous View Control</em>, Chen et al.   <a href="https://arxiv.org/abs/1901.01880">link</a></td> </tr>

</table>
<hr>
<br>
  		  <table align=center width=900px>
  			  <tr>
  	              <td width=400px>
  					<left>
				This project was partially funded by the BMBF project VIDETE(01IW1800). <br>
				We thank the SDS department at DFKI Kaiserslautern, for their support with GPU infrastructure. <br>
				The template for this website is borrowed from <a href="https://richzhang.github.io/">Richard Zhang</a>.


			</left>
		</td>
			 </tr>
		</table>

		<br>


</body>
</html>


```
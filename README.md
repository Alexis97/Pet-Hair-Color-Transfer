# Pet-Hair-Color-Transfer
## Introduction
* Train a CycleGAN to do pet hair color transfer with pet dataset (from ImageNet).
* The CycleGAN structure is from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). We made several modification on it to satisfy our transfer task.
* The train dataset is gathered from ImageNet and Google, but it is unaviliable to publish now. 
* Here are some results:
<center>
	<img src="https://github.com/Alexis97/Pet-Hair-Color-Transfer/blob/master/demos/orange2white1.png" height = 200>
	Transfer dog images from hair color of <font color="orange"> orange to <color/white> white
</center>
<center>
	<img src="https://github.com/Alexis97/Pet-Hair-Color-Transfer/blob/master/demos/white2orange1.png" height = 200>
	Transfer dog images from hair color of <color/orange> white to <color/white> orange
</center>
	
## Network Structure
* Here is the schematic diagram of our method:
<center>
	<img src="https://github.com/Alexis97/Pet-Hair-Color-Transfer/blob/master/demos/schematicDiagram.png" height = 400>
</center>

> The left column images are real pet images after segmentation which are categorized by hair color into two domains: X and Y . The middle column images are fake images generated by our generators (G1 and G2 ) which are restricted by discriminators (D2 and D1 ) to obtain a similar distribution as real images in the target domains: Y and X. The right column images are reconstructed images generated by G2 and G1 back to the original domains: X and Y . The reconstructed images are encouraged to be as similar as possible with real images by reconstruction loss.
* Here is the proposed framework of our method:
<center>
	<img src="https://github.com/Alexis97/Pet-Hair-Color-Transfer/blob/master/demos/proposedFramework-2.png" height = 400>
</center>
>Using segmentation network, an original image I is firstly masked and then resized to 256 x 256. Generator G1 takes resized I and its mask M as input and generates fake image I0. Generator G2 takes fake image I0 with mask M as input and generates reconstructed image Ic. Generator G2 also takes real image I0 with mask M as input and generates identical image Ii. Discriminator tries to distinguish I0 from images in domain Y which leads to adversarial loss Ladv. Reconstruction loss L_rec measures the pixel level distance between original image I and reconstructed one Ic by L1 norm. Identity loss Lidt measures the pixel level distance between original image I and identical one I_i by L1 norm. 

# HydroPML for large-scale flood modeling and forecast
Large-scale hydrodynamic models generally rely on fixed-resolution spatial grids and model parameters as well as incurring a high computational cost. This limits their ability to accurately forecast flood crests and issue time-critical hazard warnings. In this work, we build a fast, stable, accurate, resolution-invariant, and geometry-adaptative flood modeling and forecasting framework that can perform at large scales, namely FloodCast. The framework comprises two main modules: multi-satellite observation and hydrodynamic modeling. In the multi-satellite observation module, a real-time unsupervised change detection method and a rainfall processing and analysis tool are proposed to harness the full potential of multi-satellite observations in large-scale flood prediction. In the hydrodynamic modeling module, a  geometry-adaptive physics-informed neural solver (GeoPINS) is proposed based on the advantages of no training data in physics-informed neural networks (PINNs), as well as possessing a fast, accurate, and resolution-invariant architecture through the implementation of Fourier neural operators. To adapt to complex river geometries, we reformulate PINNs in a geometry-adaptive space, leveraging coordinate transformations and efficient numerical methods for spatial gradient solutions. GeoPINS demonstrates impressive performance on popular partial differential equations across regular and irregular domains. Building upon GeoPINS, we propose a sequence-to-sequence GeoPINS model to handle long-term temporal series and extensive spatial domains in large-scale flood modeling. This model employs sequence-to-sequence learning and hard-encoding of boundary conditions. Next, we establish a benchmark dataset in the 2022 Pakistan flood using a widely accepted finite difference numerical solution technique to assess various flood prediction methods. Finally, we validate the model in three dimensions - flood inundation range, depth, and transferability of spatiotemporal downscaling - utilizing SAR-based flood data, traditional hydrodynamic benchmarks, and concurrent optical remote sensing images. Traditional hydrodynamics and sequence-to-sequence GeoPINS exhibit exceptional agreement during high water levels, while comparative assessments with SAR-based flood depth data show that sequence-to-sequence GeoPINS outperforms traditional hydrodynamics, with smaller prediction errors. The experimental results for the Pakistan flood in 2022 indicate that the proposed method can maintain high-precision large-scale flood dynamics solutions and flood hazards can be forecast in real-time with the aid of reliable precipitation data.
# Model: Sequence-to-sequence Geometry-adaptive physics-informed neural solver (PiML)
![Geometry-adaptive physics-informed neural solver](https://github.com/HydroPML/FloodCast/blob/main/Figures/fig2.jpg)
# Study Area: 2022 Pakistan Flood
![Study area](https://github.com/HydroPML/FloodCast/blob/main/Figures/fig15.jpg)
Flood events are recurrent phenomena in Pakistan, primarily driven by intense summer monsoon rainfall and occasional tropical cyclones. In the summer monsoon season of 2022, Pakistan experienced a devastating flood event. This flood event impacted approximately one-third of Pakistan's population, resulting in the displacement of around 32 million individuals and causing the loss of 1,486 lives, including 530 children. The economic toll of this disaster has been estimated at exceeding 30 billion. Beyond the immediate consequences, the widespread destruction of agricultural fields has raised concerns of potential famine, and there is a looming threat of disease outbreaks in temporary shelters.
The study area encompasses the regions in Pakistan most severely affected by the flood, spanning the southern provinces of Punjab, Sindh, and Balochistan, covering a total land area of 85,616.5 square kilometers. The Indus River basin, a critical drainage system, plays a pivotal role in this study area's hydrology. 
# Results
**Comparison of the average depth of the study area calculated using the traditional hydrodynamics method and the average depth computed by HydroPML over a 14-day period of rainfall**  
![Comparison of the average depth of the study area calculated using the traditional hydrodynamics method and the average depth computed by HydroPML over a 14-day period of rainfall](https://github.com/HydroPML/FloodCast/blob/main/Figures/fig12.jpg)
# References
Xu, Q., Shi, Y., Bamber, J., Ouyang, C., & Zhu, X. X. (2023). A large-scale flood modeling using geometry-adaptive physics-informed neural solver and Earth observation data (No. EGU23-3276).  
Xu, Q., Shi, Y., Bamber, J., Ouyang, C., & Zhu, X. X. (2024). Large-scale flood modeling and forecasting with FloodCast (under review).
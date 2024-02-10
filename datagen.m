clear

TX = randi([0 1], 1, 1000);
RX_ideal = awgn(repmat([0 0],1, 500),30,'measured');
H1 = [1 0.07 0.5 0.8];
H2 = [1 -0.4 -0.66 0.7];
SI = [filter(H1, 1, TX(1:1:500)),filter(H2, 1, TX(501:1:1000))];
SI = (SI.^2)/5;
RX = RX_ideal + SI

save('data.mat','RX','TX','RX_ideal')



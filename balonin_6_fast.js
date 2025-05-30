// VERSION: 2025-05-30 00:30:53
// Поиск для n=6

n=6; example(n,0); 


puts("a=["+a+"];"); puts("b=["+b+"];"); 
H=twocircul(a,b); {{I=H'*H}} putm(I);
plotm(H,'XR',140,20);

function example(n, k) {
    if (n == 6) {
        if (k == 0) {
            a = [-1,-1,1];
            b = [-1,-1,1];
        }
        if (k == 1) {
            a = [-1,1,-1];
            b = [-1,1,-1];
        }
        if (k == 2) {
            a = [-1,1,1];
            b = [-1,1,1];
        }
    }

}

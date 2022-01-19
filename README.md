#   N e m a t i c s   c o d e s  
  
 E x a m p l e s   o f   f u n c t i o n   u s e .   T h e   e x a m p l e s   o f   t h e   c o d e s   l o c a t e d   i n   ` . \ n e m a t i c s \ _ b a c t e r i a \ g e t _ d e f e c t s _ f r o m _ a n g l e . m `  
  
 # #   v i s u a l i z e   n e m a t i c   f i e l d   o n   t o p   o f   r a w   i m a g e  
 	 % %   L o a d   r a w   i m a g e   c o r r e s p o n i n g   t o   a n g u l a r   m a p  
 	 f i l e p a t h   =   ' . \ e x a m p l e _ i m a g e s \ o r i e n t \ O r i e n t _ 1 _ X 1 . t i f '  
 	 A n g   =   i m r e a d ( f i l e p a t h ) ;  
 	 [ f , n a m e , e x t ]   =   f i l e p a r t s ( f i l e p a t h ) ;  
 	 d i r _ i n f o     =   d i r ( [ ' . \ e x a m p l e _ i m a g e s \ r a w \ * ' , n a m e ( 8 : e n d ) , ' . t i f ' ] ) ;  
 	 r a w _ i m g _ p a t h   =   [ d i r _ i n f o . f o l d e r   ' \ '   d i r _ i n f o . n a m e ] ;  
 	 i m s h o w ( i m r e a d ( r a w _ i m g _ p a t h ) ) ;   h o l d   o n  
 	 p l o t _ n e m a t i c _ f i e l d ( A n g ) ;  
  
 # #   o r d e r   p a r a m e t e r  
 	 % %   O r d e r   p a r a m e t e r   m a p   f r o m   a n g u l a r   m a p  
 	 f i l e p a t h   =   ' . \ e x a m p l e _ i m a g e s \ o r i e n t \ O r i e n t _ 1 _ X 1 . t i f '  
 	 A n g   =   i m r e a d ( f i l e p a t h ) ;  
 	 q q   =   o r d e r _ p a r a m e t e r ( A n g , 1 0 , 3 ) ;  
 	 i m s h o w ( q q ) ;   h o l d   o n  
  
  
 # #   d e f e c t   d e t e c t i o n  
 	 % %   F i n d   d e f e c t s   f r o m   o r d e r   a n g u l a r   m a p   ( p l o t t e d   o n   o f   o r d e r   p a r a m e t e r )  
 	 f i l e p a t h   =   ' . \ e x a m p l e _ i m a g e s \ o r i e n t \ O r i e n t _ 1 _ X 1 . t i f '  
 	 A n g   =   i m r e a d ( f i l e p a t h ) ;  
 	 q q   =   o r d e r _ p a r a m e t e r ( A n g , 1 0 , 3 ) ;  
 	 i m s h o w ( q q ) ;   h o l d   o n  
 	 [ x f ,   y f ]   =   d e t e c t D e f e c t s F r o m A n g l e ( A n g ) ;  
 	 s c a t t e r ( x f ,   y f ,   " f i l l e d " )  
  
  
 # #   [ + 1 / 2 ,   - 1 / 2 ]   d e f e c t   c l a s s i f i c a t i o n  
 	 % %   C l a s s i f i c a t i o n   o f   + 1 / 2   a n d   - 1 / 2   d e f e c t s  
 	 f i l e p a t h   =   ' . \ e x a m p l e _ i m a g e s \ o r i e n t \ O r i e n t _ 1 _ X 1 . t i f '  
 	 A n g   =   i m r e a d ( f i l e p a t h ) ;  
 	 [ f , n a m e , e x t ]   =   f i l e p a r t s ( f i l e p a t h ) ;  
 	 d i r _ i n f o     =   d i r ( [ ' . \ e x a m p l e _ i m a g e s \ r a w \ * ' , n a m e ( 8 : e n d ) , ' . t i f ' ] ) ;  
 	 r a w _ i m g _ p a t h   =   [ d i r _ i n f o . f o l d e r   ' \ '   d i r _ i n f o . n a m e ] ;  
 	 i m s h o w ( i m r e a d ( r a w _ i m g _ p a t h ) ) ;   h o l d   o n  
 	 p l o t _ n e m a t i c _ f i e l d ( A n g ) ;  
  
 	 %   D i s p l a y   d e f e c t s  
 	 [ p s _ x ,   p s _ y ,   p l o c P s i _ v e c ,   n s _ x ,   n s _ y ,   n l o c P s i _ v e c ]   =   . . .  
 	         f u n _ g e t _ p n _ D e f e c t s _ n e w D e f e c t A n g l e ( A n g ) ;  
 	 %                   f u n _ g e t _ p n _ D e f e c t s _ n e w D e f e c t A n g l e _ b l o c k p r o c ( A n g ) ;  
  
 	 p l o t _ p d e f e c t _ n d e f e c t ( p s _ x ,   p s _ y ,   p l o c P s i _ v e c , . . .  
 	         n s _ x ,   n s _ y ,   n l o c P s i _ v e c ) ;  
 	 %           f u n _ d e f e c t D r a w ( p s _ x ,   p s _ y ,   p l o c P s i _ v e c ) ;  
  
  
 
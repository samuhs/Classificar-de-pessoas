import cv2
import numpy as np 
import sys

class camera():
    
    def __init__(self,nome):
        self.nome= nome
        self.count=1

    def save_img(self):

        print("Salve 15 imagens, para isso aperte a letra S\n cada clique equivale a 1 imagem!")
        print("Ao receber a msg q foram salvar as 15 img segure/aperte Q para sair!")

        captura = cv2.VideoCapture(0) # endereço da webcam

        while(1):
            ret, frame = captura.read() # pega o VideoCapture

    
            cv2.imshow("Video", frame)#mostra o video

            if cv2.waitKey(1) & 0xFF == ord('s'):
                if self.count == 16:
                    self.count = 1
                    print("As 15 imagens foram salvas com sucesso!")
                    continue

                #salvando img 
                cv2.imwrite('process/'+ self.nome + '-' + str(self.count) + '.jpg', frame)
                print("Imagem %d salva!"%(self.count))
                self.count = self.count+1

            if cv2.waitKey(1) & 0xFF == ord('q'): # para sair as vezes tem q fica clicando Q
                break

        captura.release()
        cv2.destroyAllWindows()

#teste= camera('samuel')
#teste.save_img()
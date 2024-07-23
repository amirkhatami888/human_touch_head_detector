import os; os .system('cls')
import winsound
from matplotlib import  pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from time import gmtime, strftime ,sleep


class base_logger:
    def __init__(self,msg):
        self.LOGGER_TERMINAL_DISPLAY = True 
        self.LOGGER_FILE_DISPLAY = True 
        self.LOGGER_FILE_PATH = "log.txt"
        self.msg=msg
        self.time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        if self.LOGGER_TERMINAL_DISPLAY:
            self.logger_terminal()
        if self.LOGGER_FILE_DISPLAY:
            self.logger_file()
               
    def  logger_terminal(self):
        print(self.msg)

    def  logger_file(self):
        with open(self.LOGGER_FILE_PATH, 'a') as f:
            f.write(f"\t<<< {self.time} >>>\t" + '\n')
            f.write(self.msg + '\n')



class croper_person(object):
    def __init__(self,image_path):    
        self.image_path=image_path
        self.model_path=r'resource\lite2-detection-default.tflite'
        self.input_size = 448 
        self.thresh=0.5
        self.ratio_tol=0.05
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_image = self.image_loade_croper()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_image.numpy())
        self.interpreter.invoke()
        self.boxes = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.classes = self.interpreter.get_tensor(self.output_details[1]['index'])
        self.scores = self.interpreter.get_tensor(self.output_details[2]['index'])
        self.num_detections = self.interpreter.get_tensor(self.output_details[3]['index'])
        self.labels = pd.read_csv(r'resource\labels.csv',sep=';',index_col='ID')
        self.labels = self.labels['OBJECT (2017 REL.)'].values
        self.classes=[int(i) for i in list(self.classes[0])]
        self.pred_labels = [self.labels[i] for i in self.classes]
        self.pred_boxes = self.boxes[0]
        self.pred_scores = self.scores[0]
        self.peson_data=self.personFinder()
        
    def personFinder(self):
        self.peson_data=dict()  
        for i in range(len(self.pred_labels)):
            if self.pred_labels[i] =="person" and self.pred_scores[i]>self.thresh:
                if i==1:
                    continue
                self.peson_data.update({'score':self.pred_scores[i]})
                self.peson_data.update({'box':list(self.box_maker(self.pred_boxes[i]))
                                })
                break
        return self.peson_data
        
    def image_loade_croper(self):
        if type(self.image_path)==str:
            image = cv2.imread( self.image_path)
        else:
            image = np.array(self.image_path, dtype=np.uint8)
            
        image = cv2.resize(image, (self.input_size , self.input_size ))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = tf.convert_to_tensor(image, dtype=tf.uint8)
        image = tf.expand_dims(image , 0)
        image=  tf.cast(image, dtype=tf.uint8)
        return image
    def box_maker(self,li):
        return [int(np.round(li[0]*self.input_size)),int(np.round(li[1]*self.input_size)),int(np.round(li[2]*self.input_size)),int(np.round(li[3]*self.input_size))]
    
    def croper(self):
        if type(self.image_path)==str:
            img = cv2.imread(self.image_path)
        else:
            img = np.array(self.image_path, dtype=np.uint8)
        img=cv2.resize(img,(512,512))

        box=self.peson_data['box']
        ratio=self.ratio_tol
        tol=int(512*ratio)
        box[0]=box[0]-tol
        box[1]=box[0]-tol
        box[2]=box[2]+tol
        box[3]=box[3]+tol
        for i in range(len(box)):
            if box[i]<0:
                box[i]=0
            if box[i]>512:
                box[i]=512
        crop_img = img[box[0]:box[2], box[1]:box[3]]
        
        crop_img=np.array(crop_img,dtype=np.uint8)
        cv2.imwrite("crop_img.JPG", crop_img)
        return crop_img
    

class personFinder(object):
    def __init__(self,image_path):
        self.image_path=image_path
        self.image=self.image_loade_finder(self.image_path)
        self.model_path=r'resource\movenet-tflite-singlepose-thunder-tflite-float16-v1.tflite'
        self.size = 256 
        self.thresh=0.2
        self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()
        
        self.input_image = tf.cast(self.image, dtype=tf.uint8)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.interpreter.set_tensor(self.input_details[0]['index'], self.input_image.numpy())
        self.interpreter.invoke()
        self.keypoints_with_scores = self.interpreter.get_tensor(self.output_details[0]['index'])
        self.KeypointAbsolute=self.KeypointAbsolutefunc()
        self.box_head,self.center_head=self.head_boxfnc()
        self.point_right_wrist,self.point_left_wrist,self.rightArm,self.leftArm=self.arm_func()
        self.prob=0
         
    def image_loade_finder(self,img):
        img=croper_person(img).croper()
        image = tf.convert_to_tensor(img, dtype=tf.uint8)
        input_image = tf.expand_dims(image, axis=0)
        return tf.image.resize_with_pad(input_image, 256, 256)

    def KeypointAbsolutefunc(self):
        kpts_x=np.array(256.0  * np.array(self.keypoints_with_scores[0, 0, :, 1]).reshape(-1,1),dtype=np.uint8)
        kpts_y=np.array(256.0  * np.array(self.keypoints_with_scores[0, 0, :, 0]).reshape(-1,1),dtype=np.uint8)
        kpts_scores = np.array(self.keypoints_with_scores[0, 0, :, 2],dtype=np.float32).reshape(-1,1)
        return np.concatenate([kpts_x,kpts_y,kpts_scores],axis=1)

    def head_boxfnc(self):
        try:
            #head part
            head_key=self.KeypointAbsolute[0:5]
            score_validation=[[x,y,s] for x,y,s in head_key if s > self.thresh]

            center_head_x=sum([x*s for x,y,s in score_validation])/sum([s for x,y,s in score_validation])
            center_head_y=sum([y*s for x,y,s in score_validation])/sum([s for x,y,s in score_validation])
            center_head=[center_head_x,center_head_y]

            sorted_x=np.array(sorted(np.array(score_validation),key=lambda x:x[0]))
            diif_x=abs(sorted_x[-1][0]-sorted_x[0][0])
            sorted_y=np.array(sorted(np.array(score_validation),key=lambda x:x[1]))
            diif_y=sorted_y[-1][1]-sorted_y[0][1]

            diif_max=max(diif_x,diif_y)
            top_head_x=center_head[0]-diif_max
            top_head_y=center_head[1]-diif_max
            bottom_head_x=center_head[0]+diif_max
            bottom_head_y=center_head[1]+diif_max
            box_head=[top_head_x,top_head_y,bottom_head_x,bottom_head_y]
            return box_head,center_head
        except:
            msg=f"[!]ERROR IN head_boxfnc METHOD .not found head"
            base_logger(msg)   
            return [],[]
    
    def arm_func(self):
        point_left_shoulder = self.KeypointAbsolute[5]
        point_right_shoulder= self.KeypointAbsolute[6]
        point_left_elbow    = self.KeypointAbsolute[7]
        point_right_elbow   = self.KeypointAbsolute[8]
        point_left_wrist    = self.KeypointAbsolute[9]
        point_right_wrist   = self.KeypointAbsolute[10]
        
        rightArm=[point_right_shoulder,point_right_elbow,point_right_wrist]
        rightArm_validation=[x for x in rightArm if x[2] > self.thresh]
        rightArm_validation=sorted(rightArm_validation,key=lambda x:x[0])
        rightArm=[point_right_shoulder,point_right_elbow,point_right_wrist]
        leftArm=[point_left_shoulder,point_left_elbow,point_left_wrist]
        leftArm_validation=[x for x in leftArm if x[2] > self.thresh]
        leftArm_validation=sorted(leftArm_validation,key=lambda x:x[0])
        return point_right_wrist,point_left_wrist,rightArm_validation,leftArm_validation
        
    def Analysis_certainty(self):
        if self.isSide(self.point_right_wrist[0],self.point_right_wrist[1],self.box_head):
            self.prob=max(self.prob, self.point_right_wrist[2])
        
        elif self.isSide(self.point_left_wrist[0],self.point_left_wrist[1],self.box_head):
            self.prob=max(self.prob,self.point_left_wrist[2])

        if self.prob > self.thresh:
            return True
        else:
            return False
        
    def Analysis_maybe(self):
        for Rpoint in self.rightArm:
            if self.isSide(Rpoint[0],Rpoint[1],self.box_head):
                self.prob=max(self.prob, Rpoint[2])
            else:
                self.prob=self.prob
        for Lpoint in self.leftArm:
            if self.isSide(Lpoint[0],Lpoint[1],self.box_head):
                self.prob=max(self.prob, Lpoint[2])
            else:
                self.prob=self.prob
        

        if self.prob > self.thresh:
            return True
        else:
            return False
        
    def Analysis(self):
        if self.Analysis_certainty():
            return True,self.prob
        elif self.Analysis_maybe():
            return True,self.prob
        else:
            return False,None


    def imgage_draw(self):
        li_box= self.box_head
        center_head=self.center_head
        image_path=r"crop_img.JPG"
        img =image=cv2.imread(image_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img=cv2.resize(img,(256,256))
        plt.imshow(img)
        try:
            plt.scatter(center_head[0],center_head[1],marker='o',color='red')
            plt.gca().add_patch(
            plt.Rectangle((li_box[0], li_box[1]), li_box[2]-li_box[0], li_box[3]-li_box[1], fill=False, edgecolor='red', linewidth=2))
        except:
            pass
        
        try:
            try:
                plt.scatter(self.point_right_wrist[0],self.point_right_wrist[1],marker='o',color='blue')
            except:
                pass
            try:
                plt.scatter(self.point_left_wrist[0],self.point_left_wrist[1],marker='o',color='blue')
            except:
                pass


        except:
            pass
        try:    

            try:
                try:
                    for Rpoint in self.rightArm:
                        plt.scatter(self.Rpoint[0],self.Rpoint[1],marker='o',color='green')
                except:
                    pass
                try:
                    for Lpoint in self.leftArm:
                        plt.scatter(self.Lpoint[0],self.Lpoint[1],marker='o',color='green') 
                except:
                    pass

            except:
                pass
            
        except:
            pass
        plt.show()
    @staticmethod
    def isSide(X,Y,box_list):
        try:
            Xs=[box_list[0],box_list[2]]
            max_X=max(Xs)
            min_X=min(Xs)
            Yx=[box_list[1],box_list[3]]
            max_Y=max(Yx)
            min_Y=min(Yx)
            
            cond_x=X>min_X and X<max_X
            cond_y=Y>min_Y and Y<max_Y
            
            if cond_x and cond_y:
                return True
            else:
                return False
        except:
            msg=f"[!]ERROR IN isSide METHOD .  box_list is {box_list}"
            base_logger(msg)   
            return False

        
        
        


def get_pic():
    vid = cv2.VideoCapture(0)
    time=strftime("%Y-%m-%d %H:%M:%S", gmtime())
    ret, frame = vid.read()
    cv2.destroyAllWindows() 
    if ret == True:
        return time, frame
    else:
        return time , None   


def beeb(grade):
    winsound.Beep(int(900+grade*2000), 500)
    winsound.Beep(int(900+grade*2000), 500)
    winsound.Beep(int(900+grade*2000), 500)
    winsound.Beep(int(900+grade*2000), 500)
    winsound.Beep(int(900+grade*2000), 500)



while(True):
    time, frame=get_pic()
    if not frame is None:
        test=personFinder(frame)
        ans,prob=test.Analysis()
        
        if not ans is False:
            base_logger(f"[+]head touch detected with probability of  {prob*100}% , at time {time}")   
            beeb(prob)
            test.imgage_draw()
            
    

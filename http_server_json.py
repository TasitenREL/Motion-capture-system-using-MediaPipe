import http.server as s
import json
import datetime

debug = False

#########################################
# Webサーバ
class MyHandler(s.BaseHTTPRequestHandler):
    tracker = {"pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]} # トラッカー情報,0~2番目はxyz位置座標,3~5番目はxyz回転座標
    human_pose = {"info": [0,0,0,0,0,0,0,0]} # 姿勢

    # ログを出力させたいときはコメントアウトせよ
    #def log_message(self, format, *args):
        #return

    #---------------------------------------------------------------------
    def do_GET(self): # HTTPクライアントから GET されたときの処理（ 今回，このメソッドは消しても問題ない）
        msg="<HTTP><TITLE>TAKAGO LAB</TITLE><BODY><HR>...(o_o) JSON TEST SERVER<HR></BODY></HTML>\n".encode("utf-8")    
        # ヘッダにメッセージを付けて返す
        self.send_response(200)
        self.send_header("Content-type", 'text/html')
        self.end_headers()
        self.wfile.write(msg)

    #---------------------------------------------------------------------
    def do_POST(self): # HTTPクライアントから POST されたときの処理

        # クライアントからのリクエストの取り出し
        content_len  = int(self.headers.get("content-length")) # ヘッダからデータ長を抜き出し
        recv_body = self.rfile.read(content_len).decode('utf-8') # メッセージ取り出し

        if debug:
      	      	print("=====================")
      	      	print("recev:" + recv_body)#受信メッセージ表示
      	      	print("=====================")
      	
        print('\r'+recv_body+'                        ',end='')
      	
        recv_dic = json.loads(recv_body) # Pythonで処理しやすいように，JSONテキスト → 辞書 に変換
        send_dic = {"RETURN":"err"}    # 辞書の作成
 
        # 必要に応じて変更（----ココカラ----） 
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if "getval" in recv_dic.keys():  # 左記のキーを見つけたら
            k = recv_dic["getval"]      
            matching_values = {
                        "now" : str(datetime.datetime.now()),  # 時刻
                        "today" : str(datetime.date.today()), # 日付
                        "tracker" : MyHandler.tracker,        # トラッカー座標
                        "human_pose" : MyHandler.human_pose  # 人の姿勢
            }
            send_dic["RETURN"] = matching_values.get( k, "err.getval")
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if "put_tracker" in recv_dic.keys(): # 左記のキーを見つけたら
            if 'pos' in recv_dic["put_tracker"]: # その値を抜き出して，更に'pos'をキーに抜き出して，値を差し替え
                MyHandler.tracker = recv_dic["put_tracker"]["pos"]
                send_dic["RETURN"] = "success.put_tracker"
            else:
                send_dic["RETURN"] = "err.put_tracker"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if "put_human_pose" in recv_dic.keys(): # 左記のキーを見つけたら
            if 'info' in recv_dic["put_human_pose"]: # その値を抜き出して，更に'info'をキーに抜き出，値を差し替え
                MyHandler.human_pose = recv_dic["put_human_pose"]["info"]
                send_dic["RETURN"] = "success.put_human_pose"
            else:
                send_dic["RETURN"] = "err.put_humanpose"
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 必要に応じて変更（----ココマデ----）
        send_body = json.dumps(   # 辞書テキスト→JSON
                send_dic,         # 変換対象の辞書
                sort_keys=False,  # ソートするか
                indent=4,         # インデント（みやすさアップ）
                ensure_ascii=False)  # UTF8を含められるように
        send_body +="\n" # 見やすいように改行も付け足す


        # クライアントにレスポンスを返す
        self.send_response(200)
        self.send_header('Content-type', 'application/json;charset=utf-8')
        self.end_headers()
        if debug:
      	      	print("send:" + send_body)#送信メッセージ表示
      	      	print("=====================")
        self.wfile.write(send_body.encode("utf-8")) # メッセージの送信


#########################################
# サーバ起動
httpd = s.HTTPServer(('0.0.0.0', 50505), MyHandler)
httpd.serve_forever()

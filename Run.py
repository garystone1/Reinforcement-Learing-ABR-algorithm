if __name__ == "__main__":
    if(sys.argv[1]=="all"):
        video_traces = [
            'AsianCup_China_Uzbekistan',
            'Fengtimo_2018_11_3', 
            'game', 
            'room', 
            'sports', 
            'YYF_2018_08_12'
        ]
        netwrok_traces = [
            'fixed',
            'low',
            'medium',
            'high'
        ]
    else:
        video_traces = [sys.argv[1]]
        netwrok_traces = [sys.argv[2]]
    debug = False
    traincases = []
    for video_trace in video_traces:
        for netwrok_trace in netwrok_traces:
            traincases.append([video_trace, netwrok_trace, debug])
    N = mp.cpu_count()
    with mp.Pool(processes=N) as p:
        results = p.map(train,traincases)
    print(results)
    print("score: ", np.mean(results ,axis = 0))

import React from 'react';
import JokeList from './JokeList';
import '../css/lists.css';
import Drawer from 'material-ui/Drawer';

export default class Lists extends React.PureComponent {
    constructor(props) {
        super(props);
        const len = this.props.jokes.length;
        this.idMap = {};
        this.topicJokes = {};
        for (let i = 0; i < len; i++) {
            const jk = this.props.jokes[i];
            this.idMap[jk.id] = jk
        }
        this.state = {
            sidebarOpen: true
        }
        this.checkCache(props);
    }

    // R screws up lists with one element.
    fix(list) {
        return list ? (list.map ? list : [list]) : [];
    }

    collectJokesForTopic(topics, topic) {

        const words = topics[topic].words;
        const list = []
        const map = {}
        for (let i = 0; i < words.length; i++) {
            const ids = this.fix(words[i].joke_ids);
            for (let j = 0; j < ids.length; j++) {
                if (!map[ids[j]]) {
                    map[ids[j]] = true;
                    list.push(this.idMap[ids[j]]);
                }
            }
        }
        this.topicJokes[topic] = list.slice(0, 500);
    }

    componentWillReceiveProps(nextProps) {
        this.checkCache(nextProps);
    }

    checkCache(nextProps) {
        if (!this.topicJokes[nextProps.topicA]) this.collectJokesForTopic(nextProps.topics, nextProps.topicA);
        if (!this.topicJokes[nextProps.topicB]) this.collectJokesForTopic(nextProps.topics, nextProps.topicB);
    }

    render() {
        const topicA = {
            border: '2px solid ' + this.props.topicA_color
        }
        const topicB = {
            border: '2px solid ' + this.props.topicB_color
        }
        let sidebar = null;
        if (this.props.bar_selection.open) {
            const selection = this.props.bar_selection;
            //const color = this.props[selection.topic + '_color'];
            const topic = this.props[selection.topic];
            const ids = this.fix(this.props.topics[topic].words[selection.index].joke_ids);
            const data = ids.map(id => this.idMap[id])
            sidebar = (
                <div className='sidebar_content'>
                    <div className='sidebar_title'>
                        <h1 className='white'>"{selection.word.name}"...</h1>
                        ... has an affinity of {selection.word.p_topic} towards Topic-{topic + 1}, and occurs {selection.word.count} times.
                        <br />
                        <br />
                        Here are some jokes containing "{selection.word.name}":
                    </div>
                    <div className='sidebar_body'>
                        <JokeList
                            color={this.props.topicA_color}
                            data={data}
                        />
                    </div>
                </div>
            )
        }
        return (
            <div>
                <Drawer open={this.props.bar_selection.open}
                    anchor='right'
                    onRequestClose={this.props.clearSelection}>
                    {sidebar}
                </Drawer>
                <div className='lists_title'>
                    <span style={topicA}>Jokes in Topic {this.props.topicA + 1}</span>
                    <span style={topicB}>Jokes in Topic {this.props.topicB + 1}</span>
                </div>
                <div className='lists_body'>
                    <div className='list_container'>
                        <JokeList
                            color={this.props.topicA_color}
                            data={this.topicJokes[this.props.topicA]}
                        />
                    </div>
                    <div className='list_container'>
                        <JokeList
                            color={this.props.topicB_color}
                            data={this.topicJokes[this.props.topicB]}
                        />
                    </div>
                </div>
            </div>
        )
    }
}
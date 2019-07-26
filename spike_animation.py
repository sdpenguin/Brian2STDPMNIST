import numpy as np
import matplotlib
matplotlib.use('PDF')
from matplotlib import pyplot as plt
from matplotlib import animation, rc
import brian2 as b2

rc('animation', html='html5')


def spike_animation(spikes,
                    total_example_time,
                    imgsize=28,
                    example_number=None,
                    gif_filename='spikes_example',
                    max_counts=50):
    if example_number is None:
        example_number = np.random.choice(spikes['Xe']['tbin'])
    spikes_ex = {p: spikes[p].set_index('tbin').loc[example_number] for p in spikes}
    dtexbin = 5 * b2.ms
    for p in spikes:
        spikes_ex[p]['texbin'] = ((spikes_ex[p]['t'] - spikes_ex[p]['t'].min()) / 50).astype(np.int)
    counts_ex = spikes_ex['Xe'].groupby(['texbin', 'x', 'y'])['t'].count().reset_index(level=['x', 'y'])

    tott = total_example_time / dtexbin
    times = np.arange(tott)

    fig, (axX, axA) = plt.subplots(1, 2, figsize=(15, 5))
    img = axX.imshow(np.zeros((imgsize, imgsize)),
                     vmin=0, vmax=max_counts)
    dotsX, = axX.plot([], [], 'r.')
    title = axX.set_title('', loc='right')
    axX.axis('off')

    n_e = spikes['Ae']['i'].max()
    dotsAe, = axA.plot([], [], 'r.')
    dotsAi, = axA.plot([], [], 'bo', mfc='none')
    axA.set_xlim(-1, tott * dtexbin / b2.ms)
    axA.set_ylim(-1, n_e)
    axA.set_xlabel('$t$ (ms)')
    axA.set_ylabel('neuron index')
    lineA, = axA.plot([], [], 'k-')
    fig.set_tight_layout(True)

    def init():
        a = img.get_array()
        a[:] *= 0
        dotsX.set_data([], [])
        dotsAe.set_data([], [])
        dotsAi.set_data([], [])
        title.set_text('')
        return [title, dotsX, dotsAe, dotsAi, lineA, img]

    def animate(t):
        artists = [title, dotsX, dotsAe, dotsAi, lineA]
        ti = dtexbin / b2.ms * (t + 1)
        title.set_text('{} ms'.format(ti))
        if t in counts_ex.index:
            artists.append(img)
            k = counts_ex.loc[t]
            a = img.get_array()
            a[k['y'], k['x']] += 1
            dotsX.set_data(k['x'], k['y'])
        else:
            dotsX.set_data([], [])

        lineA.set_data([ti, ti], [-1, n_e])

        Ai_spikes = spikes_ex['Ai'][spikes_ex['Ai']['texbin'] <= t]
        dotsAi.set_data(Ai_spikes['texbin'] * dtexbin / b2.ms, Ai_spikes['i'])
        Ae_spikes = spikes_ex['Ae'][spikes_ex['Ae']['texbin'] <= t]
        dotsAe.set_data(Ae_spikes['texbin'] * dtexbin / b2.ms, Ae_spikes['i'])
        return artists

    duration = 10000  # ms
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=times, interval=duration / len(times),
                                   repeat=False, blit=True)
    plt.close()
    if gif_filename is not None:
        anim.save('{}.gif'.format(gif_filename), writer='imagemagick', fps=1000 * len(times) / duration)
    return anim